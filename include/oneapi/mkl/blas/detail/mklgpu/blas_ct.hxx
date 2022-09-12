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

void herk(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          std::int64_t n, std::int64_t k, float alpha, sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, float beta, sycl::buffer<std::complex<float>, 1> &c,
          std::int64_t ldc) {
    herk_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::herk(selector.get_queue(), upper_lower, trans, n, k, alpha, a,
                                           lda, beta, c, ldc);
    herk_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void herk(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          std::int64_t n, std::int64_t k, double alpha,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, double beta,
          sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    herk_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::herk(selector.get_queue(), upper_lower, trans, n, k, alpha, a,
                                           lda, beta, c, ldc);
    herk_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void scal(backend_selector<backend::mklgpu> selector, std::int64_t n, float alpha,
          sycl::buffer<float, 1> &x, std::int64_t incx) {
    scal_precondition(selector.get_queue(), n, alpha, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::scal(selector.get_queue(), n, alpha, x, incx);
    scal_postcondition(selector.get_queue(), n, alpha, x, incx);
}

void scal(backend_selector<backend::mklgpu> selector, std::int64_t n, double alpha,
          sycl::buffer<double, 1> &x, std::int64_t incx) {
    scal_precondition(selector.get_queue(), n, alpha, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::scal(selector.get_queue(), n, alpha, x, incx);
    scal_postcondition(selector.get_queue(), n, alpha, x, incx);
}

void scal(backend_selector<backend::mklgpu> selector, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    scal_precondition(selector.get_queue(), n, alpha, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::scal(selector.get_queue(), n, alpha, x, incx);
    scal_postcondition(selector.get_queue(), n, alpha, x, incx);
}

void scal(backend_selector<backend::mklgpu> selector, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    scal_precondition(selector.get_queue(), n, alpha, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::scal(selector.get_queue(), n, alpha, x, incx);
    scal_postcondition(selector.get_queue(), n, alpha, x, incx);
}

void scal(backend_selector<backend::mklgpu> selector, std::int64_t n, float alpha,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    scal_precondition(selector.get_queue(), n, alpha, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::scal(selector.get_queue(), n, alpha, x, incx);
    scal_postcondition(selector.get_queue(), n, alpha, x, incx);
}

void scal(backend_selector<backend::mklgpu> selector, std::int64_t n, double alpha,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    scal_precondition(selector.get_queue(), n, alpha, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::scal(selector.get_queue(), n, alpha, x, incx);
    scal_postcondition(selector.get_queue(), n, alpha, x, incx);
}

void trmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, sycl::buffer<float, 1> &a, std::int64_t lda,
          sycl::buffer<float, 1> &x, std::int64_t incx) {
    trmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::trmv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           a, lda, x, incx);
    trmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, sycl::buffer<double, 1> &a, std::int64_t lda,
          sycl::buffer<double, 1> &x, std::int64_t incx) {
    trmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::trmv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           a, lda, x, incx);
    trmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    trmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::trmv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           a, lda, x, incx);
    trmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    trmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::trmv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           a, lda, x, incx);
    trmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void tpmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, sycl::buffer<float, 1> &a,
          sycl::buffer<float, 1> &x, std::int64_t incx) {
    tpmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::tpmv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           a, x, incx);
    tpmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, sycl::buffer<double, 1> &a,
          sycl::buffer<double, 1> &x, std::int64_t incx) {
    tpmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::tpmv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           a, x, incx);
    tpmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, sycl::buffer<std::complex<float>, 1> &a,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    tpmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::tpmv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           a, x, incx);
    tpmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, sycl::buffer<std::complex<double>, 1> &a,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    tpmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::tpmv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           a, x, incx);
    tpmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx);
}

void spr(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n, float alpha,
         sycl::buffer<float, 1> &x, std::int64_t incx, sycl::buffer<float, 1> &a) {
    spr_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a);
    oneapi::mkl::blas::mklgpu::MAJOR::spr(selector.get_queue(), upper_lower, n, alpha, x, incx, a);
    spr_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a);
}

void spr(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n, double alpha,
         sycl::buffer<double, 1> &x, std::int64_t incx, sycl::buffer<double, 1> &a) {
    spr_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a);
    oneapi::mkl::blas::mklgpu::MAJOR::spr(selector.get_queue(), upper_lower, n, alpha, x, incx, a);
    spr_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a);
}

void gemm_batch(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b, float beta,
                sycl::buffer<float, 1> &c, std::int64_t ldc, std::int64_t stride_c,
                std::int64_t batch_size) {
    gemm_batch_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                            b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::gemm_batch(selector.get_queue(), transa, transb, m, n, k,
                                                 alpha, a, lda, stride_a, b, ldb, stride_b, beta, c,
                                                 ldc, stride_c, batch_size);
    gemm_batch_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                             b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

void gemm_batch(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                double beta, sycl::buffer<double, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    gemm_batch_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                            b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::gemm_batch(selector.get_queue(), transa, transb, m, n, k,
                                                 alpha, a, lda, stride_a, b, ldb, stride_b, beta, c,
                                                 ldc, stride_c, batch_size);
    gemm_batch_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                             b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

void gemm_batch(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, sycl::buffer<std::complex<float>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    gemm_batch_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                            b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::gemm_batch(selector.get_queue(), transa, transb, m, n, k,
                                                 alpha, a, lda, stride_a, b, ldb, stride_b, beta, c,
                                                 ldc, stride_c, batch_size);
    gemm_batch_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                             b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

void gemm_batch(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, sycl::buffer<std::complex<double>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    gemm_batch_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                            b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::gemm_batch(selector.get_queue(), transa, transb, m, n, k,
                                                 alpha, a, lda, stride_a, b, ldb, stride_b, beta, c,
                                                 ldc, stride_c, batch_size);
    gemm_batch_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                             b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

void gemm_batch(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, sycl::half alpha,
                sycl::buffer<sycl::half, 1> &a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<sycl::half, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                sycl::half beta, sycl::buffer<sycl::half, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    gemm_batch_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                            b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::gemm_batch(selector.get_queue(), transa, transb, m, n, k,
                                                 alpha, a, lda, stride_a, b, ldb, stride_b, beta, c,
                                                 ldc, stride_c, batch_size);
    gemm_batch_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                             b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

void syrk(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          std::int64_t n, std::int64_t k, float alpha, sycl::buffer<float, 1> &a,
          std::int64_t lda, float beta, sycl::buffer<float, 1> &c, std::int64_t ldc) {
    syrk_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::syrk(selector.get_queue(), upper_lower, trans, n, k, alpha, a,
                                           lda, beta, c, ldc);
    syrk_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void syrk(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          std::int64_t n, std::int64_t k, double alpha, sycl::buffer<double, 1> &a,
          std::int64_t lda, double beta, sycl::buffer<double, 1> &c, std::int64_t ldc) {
    syrk_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::syrk(selector.get_queue(), upper_lower, trans, n, k, alpha, a,
                                           lda, beta, c, ldc);
    syrk_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void syrk(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          std::int64_t n, std::int64_t k, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    syrk_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::syrk(selector.get_queue(), upper_lower, trans, n, k, alpha, a,
                                           lda, beta, c, ldc);
    syrk_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void syrk(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          std::int64_t n, std::int64_t k, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    syrk_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::syrk(selector.get_queue(), upper_lower, trans, n, k, alpha, a,
                                           lda, beta, c, ldc);
    syrk_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void syrk_batch(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                std::int64_t n, std::int64_t k, float alpha, sycl::buffer<float, 1> &a,
                std::int64_t lda, std::int64_t stride_a, float beta, sycl::buffer<float, 1> &c,
                std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
    syrk_batch_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, stride_a,
                            beta, c, ldc, stride_c, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::syrk_batch(selector.get_queue(), upper_lower, trans, n, k,
                                                 alpha, a, lda, stride_a, beta, c, ldc, stride_c,
                                                 batch_size);
    syrk_batch_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda,
                             stride_a, beta, c, ldc, stride_c, batch_size);
}

void syrk_batch(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                std::int64_t n, std::int64_t k, double alpha, sycl::buffer<double, 1> &a,
                std::int64_t lda, std::int64_t stride_a, double beta,
                sycl::buffer<double, 1> &c, std::int64_t ldc, std::int64_t stride_c,
                std::int64_t batch_size) {
    syrk_batch_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, stride_a,
                            beta, c, ldc, stride_c, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::syrk_batch(selector.get_queue(), upper_lower, trans, n, k,
                                                 alpha, a, lda, stride_a, beta, c, ldc, stride_c,
                                                 batch_size);
    syrk_batch_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda,
                             stride_a, beta, c, ldc, stride_c, batch_size);
}

void syrk_batch(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                std::int64_t n, std::int64_t k, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, std::complex<float> beta,
                sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    syrk_batch_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, stride_a,
                            beta, c, ldc, stride_c, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::syrk_batch(selector.get_queue(), upper_lower, trans, n, k,
                                                 alpha, a, lda, stride_a, beta, c, ldc, stride_c,
                                                 batch_size);
    syrk_batch_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda,
                             stride_a, beta, c, ldc, stride_c, batch_size);
}

void syrk_batch(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                std::int64_t n, std::int64_t k, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    syrk_batch_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, stride_a,
                            beta, c, ldc, stride_c, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::syrk_batch(selector.get_queue(), upper_lower, trans, n, k,
                                                 alpha, a, lda, stride_a, beta, c, ldc, stride_c,
                                                 batch_size);
    syrk_batch_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda,
                             stride_a, beta, c, ldc, stride_c, batch_size);
}

void her2(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    her2_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::blas::mklgpu::MAJOR::her2(selector.get_queue(), upper_lower, n, alpha, x, incx, y,
                                           incy, a, lda);
    her2_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

void her2(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    her2_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::blas::mklgpu::MAJOR::her2(selector.get_queue(), upper_lower, n, alpha, x, incx, y,
                                           incy, a, lda);
    her2_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

void hbmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
          std::int64_t k, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &y,
          std::int64_t incy) {
    hbmv_precondition(selector.get_queue(), upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                      incy);
    oneapi::mkl::blas::mklgpu::MAJOR::hbmv(selector.get_queue(), upper_lower, n, k, alpha, a, lda,
                                           x, incx, beta, y, incy);
    hbmv_postcondition(selector.get_queue(), upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                       incy);
}

void hbmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
          std::int64_t k, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    hbmv_precondition(selector.get_queue(), upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                      incy);
    oneapi::mkl::blas::mklgpu::MAJOR::hbmv(selector.get_queue(), upper_lower, n, k, alpha, a, lda,
                                           x, incx, beta, y, incy);
    hbmv_postcondition(selector.get_queue(), upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                       incy);
}

void rot(backend_selector<backend::mklgpu> selector, std::int64_t n,
         sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy, float c, float s) {
    rot_precondition(selector.get_queue(), n, x, incx, y, incy, c, s);
    oneapi::mkl::blas::mklgpu::MAJOR::rot(selector.get_queue(), n, x, incx, y, incy, c, s);
    rot_postcondition(selector.get_queue(), n, x, incx, y, incy, c, s);
}

void rot(backend_selector<backend::mklgpu> selector, std::int64_t n,
         sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy, double c, double s) {
    rot_precondition(selector.get_queue(), n, x, incx, y, incy, c, s);
    oneapi::mkl::blas::mklgpu::MAJOR::rot(selector.get_queue(), n, x, incx, y, incy, c, s);
    rot_postcondition(selector.get_queue(), n, x, incx, y, incy, c, s);
}

void rot(backend_selector<backend::mklgpu> selector, std::int64_t n, sycl::buffer<float, 1> &x,
         std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy, float c, float s) {
    rot_precondition(selector.get_queue(), n, x, incx, y, incy, c, s);
    oneapi::mkl::blas::mklgpu::MAJOR::rot(selector.get_queue(), n, x, incx, y, incy, c, s);
    rot_postcondition(selector.get_queue(), n, x, incx, y, incy, c, s);
}

void rot(backend_selector<backend::mklgpu> selector, std::int64_t n, sycl::buffer<double, 1> &x,
         std::int64_t incx, sycl::buffer<double, 1> &y, std::int64_t incy, double c, double s) {
    rot_precondition(selector.get_queue(), n, x, incx, y, incy, c, s);
    oneapi::mkl::blas::mklgpu::MAJOR::rot(selector.get_queue(), n, x, incx, y, incy, c, s);
    rot_postcondition(selector.get_queue(), n, x, incx, y, incy, c, s);
}

void axpy(backend_selector<backend::mklgpu> selector, std::int64_t n, float alpha,
          sycl::buffer<float, 1> &x, std::int64_t incx, sycl::buffer<float, 1> &y,
          std::int64_t incy) {
    axpy_precondition(selector.get_queue(), n, alpha, x, incx, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::axpy(selector.get_queue(), n, alpha, x, incx, y, incy);
    axpy_postcondition(selector.get_queue(), n, alpha, x, incx, y, incy);
}

void axpy(backend_selector<backend::mklgpu> selector, std::int64_t n, double alpha,
          sycl::buffer<double, 1> &x, std::int64_t incx, sycl::buffer<double, 1> &y,
          std::int64_t incy) {
    axpy_precondition(selector.get_queue(), n, alpha, x, incx, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::axpy(selector.get_queue(), n, alpha, x, incx, y, incy);
    axpy_postcondition(selector.get_queue(), n, alpha, x, incx, y, incy);
}

void axpy(backend_selector<backend::mklgpu> selector, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    axpy_precondition(selector.get_queue(), n, alpha, x, incx, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::axpy(selector.get_queue(), n, alpha, x, incx, y, incy);
    axpy_postcondition(selector.get_queue(), n, alpha, x, incx, y, incy);
}

void axpy(backend_selector<backend::mklgpu> selector, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    axpy_precondition(selector.get_queue(), n, alpha, x, incx, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::axpy(selector.get_queue(), n, alpha, x, incx, y, incy);
    axpy_postcondition(selector.get_queue(), n, alpha, x, incx, y, incy);
}

void axpy_batch(backend_selector<backend::mklgpu> selector, std::int64_t n, float alpha,
                sycl::buffer<float, 1> &x, std::int64_t incx, std::int64_t stridex,
                sycl::buffer<float, 1> &y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size) {
    axpy_batch_precondition(selector.get_queue(), n, alpha, x, incx, stridex, y, incy, stridey,
                            batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::axpy_batch(selector.get_queue(), n, alpha, x, incx, stridex,
                                                 y, incy, stridey, batch_size);
    axpy_batch_postcondition(selector.get_queue(), n, alpha, x, incx, stridex, y, incy, stridey,
                             batch_size);
}

void axpy_batch(backend_selector<backend::mklgpu> selector, std::int64_t n, double alpha,
                sycl::buffer<double, 1> &x, std::int64_t incx, std::int64_t stridex,
                sycl::buffer<double, 1> &y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size) {
    axpy_batch_precondition(selector.get_queue(), n, alpha, x, incx, stridex, y, incy, stridey,
                            batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::axpy_batch(selector.get_queue(), n, alpha, x, incx, stridex,
                                                 y, incy, stridey, batch_size);
    axpy_batch_postcondition(selector.get_queue(), n, alpha, x, incx, stridex, y, incy, stridey,
                             batch_size);
}

void axpy_batch(backend_selector<backend::mklgpu> selector, std::int64_t n,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &x,
                std::int64_t incx, std::int64_t stridex,
                sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                std::int64_t stridey, std::int64_t batch_size) {
    axpy_batch_precondition(selector.get_queue(), n, alpha, x, incx, stridex, y, incy, stridey,
                            batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::axpy_batch(selector.get_queue(), n, alpha, x, incx, stridex,
                                                 y, incy, stridey, batch_size);
    axpy_batch_postcondition(selector.get_queue(), n, alpha, x, incx, stridex, y, incy, stridey,
                             batch_size);
}

void axpy_batch(backend_selector<backend::mklgpu> selector, std::int64_t n,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &x,
                std::int64_t incx, std::int64_t stridex,
                sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                std::int64_t stridey, std::int64_t batch_size) {
    axpy_batch_precondition(selector.get_queue(), n, alpha, x, incx, stridex, y, incy, stridey,
                            batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::axpy_batch(selector.get_queue(), n, alpha, x, incx, stridex,
                                                 y, incy, stridey, batch_size);
    axpy_batch_postcondition(selector.get_queue(), n, alpha, x, incx, stridex, y, incy, stridey,
                             batch_size);
}

void axpby(backend_selector<backend::mklgpu> selector, std::int64_t n, float alpha,
           sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
           sycl::buffer<float, 1> &y, std::int64_t incy) {
    axpby_precondition(selector.get_queue(), n, alpha, x, incx, beta, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::axpby(selector.get_queue(), n, alpha, x, incx, beta, y, incy);
    axpby_postcondition(selector.get_queue(), n, alpha, x, incx, beta, y, incy);
}

void axpby(backend_selector<backend::mklgpu> selector, std::int64_t n, double alpha,
           sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
           sycl::buffer<double, 1> &y, std::int64_t incy) {
    axpby_precondition(selector.get_queue(), n, alpha, x, incx, beta, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::axpby(selector.get_queue(), n, alpha, x, incx, beta, y, incy);
    axpby_postcondition(selector.get_queue(), n, alpha, x, incx, beta, y, incy);
}

void axpby(backend_selector<backend::mklgpu> selector, std::int64_t n, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    axpby_precondition(selector.get_queue(), n, alpha, x, incx, beta, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::axpby(selector.get_queue(), n, alpha, x, incx, beta, y, incy);
    axpby_postcondition(selector.get_queue(), n, alpha, x, incx, beta, y, incy);
}

void axpby(backend_selector<backend::mklgpu> selector, std::int64_t n, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
           std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &y,
           std::int64_t incy) {
    axpby_precondition(selector.get_queue(), n, alpha, x, incx, beta, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::axpby(selector.get_queue(), n, alpha, x, incx, beta, y, incy);
    axpby_postcondition(selector.get_queue(), n, alpha, x, incx, beta, y, incy);
}

void sdsdot(backend_selector<backend::mklgpu> selector, std::int64_t n, float sb,
            sycl::buffer<float, 1> &x, std::int64_t incx, sycl::buffer<float, 1> &y,
            std::int64_t incy, sycl::buffer<float, 1> &result) {
    sdsdot_precondition(selector.get_queue(), n, sb, x, incx, y, incy, result);
    oneapi::mkl::blas::mklgpu::MAJOR::sdsdot(selector.get_queue(), n, sb, x, incx, y, incy, result);
    sdsdot_postcondition(selector.get_queue(), n, sb, x, incx, y, incy, result);
}

void gerc(backend_selector<backend::mklgpu> selector, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    gerc_precondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::blas::mklgpu::MAJOR::gerc(selector.get_queue(), m, n, alpha, x, incx, y, incy, a,
                                           lda);
    gerc_postcondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda);
}

void gerc(backend_selector<backend::mklgpu> selector, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    gerc_precondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::blas::mklgpu::MAJOR::gerc(selector.get_queue(), m, n, alpha, x, incx, y, incy, a,
                                           lda);
    gerc_postcondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda);
}

void syr2k(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
           std::int64_t n, std::int64_t k, float alpha, sycl::buffer<float, 1> &a,
           std::int64_t lda, sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
           sycl::buffer<float, 1> &c, std::int64_t ldc) {
    syr2k_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::syr2k(selector.get_queue(), upper_lower, trans, n, k, alpha,
                                            a, lda, b, ldb, beta, c, ldc);
    syr2k_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                        c, ldc);
}

void syr2k(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
           std::int64_t n, std::int64_t k, double alpha, sycl::buffer<double, 1> &a,
           std::int64_t lda, sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
           sycl::buffer<double, 1> &c, std::int64_t ldc) {
    syr2k_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::syr2k(selector.get_queue(), upper_lower, trans, n, k, alpha,
                                            a, lda, b, ldb, beta, c, ldc);
    syr2k_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                        c, ldc);
}

void syr2k(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
           std::int64_t n, std::int64_t k, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    syr2k_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::syr2k(selector.get_queue(), upper_lower, trans, n, k, alpha,
                                            a, lda, b, ldb, beta, c, ldc);
    syr2k_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                        c, ldc);
}

void syr2k(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
           std::int64_t n, std::int64_t k, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc) {
    syr2k_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::syr2k(selector.get_queue(), upper_lower, trans, n, k, alpha,
                                            a, lda, b, ldb, beta, c, ldc);
    syr2k_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                        c, ldc);
}

void gemv(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
          std::int64_t n, float alpha, sycl::buffer<float, 1> &a, std::int64_t lda,
          sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          sycl::buffer<float, 1> &y, std::int64_t incy) {
    gemv_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::gemv(selector.get_queue(), trans, m, n, alpha, a, lda, x,
                                           incx, beta, y, incy);
    gemv_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
          std::int64_t n, double alpha, sycl::buffer<double, 1> &a, std::int64_t lda,
          sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          sycl::buffer<double, 1> &y, std::int64_t incy) {
    gemv_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::gemv(selector.get_queue(), trans, m, n, alpha, a, lda, x,
                                           incx, beta, y, incy);
    gemv_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
          std::int64_t n, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &y,
          std::int64_t incy) {
    gemv_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::gemv(selector.get_queue(), trans, m, n, alpha, a, lda, x,
                                           incx, beta, y, incy);
    gemv_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
          std::int64_t n, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    gemv_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::gemv(selector.get_queue(), trans, m, n, alpha, a, lda, x,
                                           incx, beta, y, incy);
    gemv_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv_batch(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                std::int64_t n, float alpha, sycl::buffer<float, 1> &a, std::int64_t lda,
                std::int64_t stridea, sycl::buffer<float, 1> &x, std::int64_t incx,
                std::int64_t stridex, float beta, sycl::buffer<float, 1> &y, std::int64_t incy,
                std::int64_t stridey, std::int64_t batch_size) {
    gemv_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx,
                            stridex, beta, y, incy, stridey, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::gemv_batch(selector.get_queue(), trans, m, n, alpha, a, lda,
                                                 stridea, x, incx, stridex, beta, y, incy, stridey,
                                                 batch_size);
    gemv_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx,
                             stridex, beta, y, incy, stridey, batch_size);
}

void gemv_batch(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                std::int64_t n, double alpha, sycl::buffer<double, 1> &a, std::int64_t lda,
                std::int64_t stridea, sycl::buffer<double, 1> &x, std::int64_t incx,
                std::int64_t stridex, double beta, sycl::buffer<double, 1> &y,
                std::int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
    gemv_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx,
                            stridex, beta, y, incy, stridey, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::gemv_batch(selector.get_queue(), trans, m, n, alpha, a, lda,
                                                 stridea, x, incx, stridex, beta, y, incy, stridey,
                                                 batch_size);
    gemv_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx,
                             stridex, beta, y, incy, stridey, batch_size);
}

void gemv_batch(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                std::int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda, std::int64_t stridea,
                sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                std::int64_t stridex, std::complex<float> beta,
                sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                std::int64_t stridey, std::int64_t batch_size) {
    gemv_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx,
                            stridex, beta, y, incy, stridey, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::gemv_batch(selector.get_queue(), trans, m, n, alpha, a, lda,
                                                 stridea, x, incx, stridex, beta, y, incy, stridey,
                                                 batch_size);
    gemv_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx,
                             stridex, beta, y, incy, stridey, batch_size);
}

void gemv_batch(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                std::int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stridea, sycl::buffer<std::complex<double>, 1> &x,
                std::int64_t incx, std::int64_t stridex, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                std::int64_t stridey, std::int64_t batch_size) {
    gemv_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx,
                            stridex, beta, y, incy, stridey, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::gemv_batch(selector.get_queue(), trans, m, n, alpha, a, lda,
                                                 stridea, x, incx, stridex, beta, y, incy, stridey,
                                                 batch_size);
    gemv_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx,
                             stridex, beta, y, incy, stridey, batch_size);
}

void dgmm_batch(backend_selector<backend::mklgpu> selector, side left_right, std::int64_t m,
                std::int64_t n, sycl::buffer<float, 1> &a, std::int64_t lda,
                std::int64_t stridea, sycl::buffer<float, 1> &x, std::int64_t incx,
                std::int64_t stridex, sycl::buffer<float, 1> &c, std::int64_t ldc,
                std::int64_t stridec, std::int64_t batch_size) {
    dgmm_batch_precondition(selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx,
                            stridex, c, ldc, stridec, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::dgmm_batch(selector.get_queue(), left_right, m, n, a, lda,
                                                 stridea, x, incx, stridex, c, ldc, stridec,
                                                 batch_size);
    dgmm_batch_postcondition(selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx,
                             stridex, c, ldc, stridec, batch_size);
}

void dgmm_batch(backend_selector<backend::mklgpu> selector, side left_right, std::int64_t m,
                std::int64_t n, sycl::buffer<double, 1> &a, std::int64_t lda,
                std::int64_t stridea, sycl::buffer<double, 1> &x, std::int64_t incx,
                std::int64_t stridex, sycl::buffer<double, 1> &c, std::int64_t ldc,
                std::int64_t stridec, std::int64_t batch_size) {
    dgmm_batch_precondition(selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx,
                            stridex, c, ldc, stridec, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::dgmm_batch(selector.get_queue(), left_right, m, n, a, lda,
                                                 stridea, x, incx, stridex, c, ldc, stridec,
                                                 batch_size);
    dgmm_batch_postcondition(selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx,
                             stridex, c, ldc, stridec, batch_size);
}

void dgmm_batch(backend_selector<backend::mklgpu> selector, side left_right, std::int64_t m,
                std::int64_t n, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stridea, sycl::buffer<std::complex<float>, 1> &x,
                std::int64_t incx, std::int64_t stridex,
                sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc, std::int64_t stridec,
                std::int64_t batch_size) {
    dgmm_batch_precondition(selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx,
                            stridex, c, ldc, stridec, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::dgmm_batch(selector.get_queue(), left_right, m, n, a, lda,
                                                 stridea, x, incx, stridex, c, ldc, stridec,
                                                 batch_size);
    dgmm_batch_postcondition(selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx,
                             stridex, c, ldc, stridec, batch_size);
}

void dgmm_batch(backend_selector<backend::mklgpu> selector, side left_right, std::int64_t m,
                std::int64_t n, sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stridea, sycl::buffer<std::complex<double>, 1> &x,
                std::int64_t incx, std::int64_t stridex,
                sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                std::int64_t stridec, std::int64_t batch_size) {
    dgmm_batch_precondition(selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx,
                            stridex, c, ldc, stridec, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::dgmm_batch(selector.get_queue(), left_right, m, n, a, lda,
                                                 stridea, x, incx, stridex, c, ldc, stridec,
                                                 batch_size);
    dgmm_batch_postcondition(selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx,
                             stridex, c, ldc, stridec, batch_size);
}

void her(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n, float alpha,
         sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    her_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, lda);
    oneapi::mkl::blas::mklgpu::MAJOR::her(selector.get_queue(), upper_lower, n, alpha, x, incx, a,
                                          lda);
    her_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, lda);
}

void her(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n, double alpha,
         sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    her_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, lda);
    oneapi::mkl::blas::mklgpu::MAJOR::her(selector.get_queue(), upper_lower, n, alpha, x, incx, a,
                                          lda);
    her_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, lda);
}

void hpr(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n, float alpha,
         sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         sycl::buffer<std::complex<float>, 1> &a) {
    hpr_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a);
    oneapi::mkl::blas::mklgpu::MAJOR::hpr(selector.get_queue(), upper_lower, n, alpha, x, incx, a);
    hpr_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a);
}

void hpr(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n, double alpha,
         sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         sycl::buffer<std::complex<double>, 1> &a) {
    hpr_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a);
    oneapi::mkl::blas::mklgpu::MAJOR::hpr(selector.get_queue(), upper_lower, n, alpha, x, incx, a);
    hpr_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a);
}

void iamin(backend_selector<backend::mklgpu> selector, std::int64_t n,
           sycl::buffer<float, 1> &x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(selector.get_queue(), n, x, incx, result);
    oneapi::mkl::blas::mklgpu::MAJOR::iamin(selector.get_queue(), n, x, incx, result);
    iamin_postcondition(selector.get_queue(), n, x, incx, result);
}

void iamin(backend_selector<backend::mklgpu> selector, std::int64_t n,
           sycl::buffer<double, 1> &x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(selector.get_queue(), n, x, incx, result);
    oneapi::mkl::blas::mklgpu::MAJOR::iamin(selector.get_queue(), n, x, incx, result);
    iamin_postcondition(selector.get_queue(), n, x, incx, result);
}

void iamin(backend_selector<backend::mklgpu> selector, std::int64_t n,
           sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(selector.get_queue(), n, x, incx, result);
    oneapi::mkl::blas::mklgpu::MAJOR::iamin(selector.get_queue(), n, x, incx, result);
    iamin_postcondition(selector.get_queue(), n, x, incx, result);
}

void iamin(backend_selector<backend::mklgpu> selector, std::int64_t n,
           sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(selector.get_queue(), n, x, incx, result);
    oneapi::mkl::blas::mklgpu::MAJOR::iamin(selector.get_queue(), n, x, incx, result);
    iamin_postcondition(selector.get_queue(), n, x, incx, result);
}

void hpmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    hpmv_precondition(selector.get_queue(), upper_lower, n, alpha, a, x, incx, beta, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::hpmv(selector.get_queue(), upper_lower, n, alpha, a, x, incx,
                                           beta, y, incy);
    hpmv_postcondition(selector.get_queue(), upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

void hpmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    hpmv_precondition(selector.get_queue(), upper_lower, n, alpha, a, x, incx, beta, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::hpmv(selector.get_queue(), upper_lower, n, alpha, a, x, incx,
                                           beta, y, incy);
    hpmv_postcondition(selector.get_queue(), upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

void spmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n, float alpha,
          sycl::buffer<float, 1> &a, sycl::buffer<float, 1> &x, std::int64_t incx,
          float beta, sycl::buffer<float, 1> &y, std::int64_t incy) {
    spmv_precondition(selector.get_queue(), upper_lower, n, alpha, a, x, incx, beta, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::spmv(selector.get_queue(), upper_lower, n, alpha, a, x, incx,
                                           beta, y, incy);
    spmv_postcondition(selector.get_queue(), upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

void spmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
          double alpha, sycl::buffer<double, 1> &a, sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, sycl::buffer<double, 1> &y, std::int64_t incy) {
    spmv_precondition(selector.get_queue(), upper_lower, n, alpha, a, x, incx, beta, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::spmv(selector.get_queue(), upper_lower, n, alpha, a, x, incx,
                                           beta, y, incy);
    spmv_postcondition(selector.get_queue(), upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

void gemm_bias(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
               offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
               sycl::buffer<int8_t, 1> &a, std::int64_t lda, int8_t ao,
               sycl::buffer<uint8_t, 1> &b, std::int64_t ldb, uint8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, std::int64_t ldc,
               sycl::buffer<int32_t, 1> &co) {
    gemm_bias_precondition(selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda,
                           ao, b, ldb, bo, beta, c, ldc, co);
    oneapi::mkl::blas::mklgpu::MAJOR::gemm_bias(selector.get_queue(), transa, transb, offsetc, m, n,
                                                k, alpha, a, lda, ao, b, ldb, bo, beta, c, ldc, co);
    gemm_bias_postcondition(selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda,
                            ao, b, ldb, bo, beta, c, ldc, co);
}

void gemm_bias(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
               offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
               sycl::buffer<int8_t, 1> &a, std::int64_t lda, int8_t ao,
               sycl::buffer<int8_t, 1> &b, std::int64_t ldb, int8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, std::int64_t ldc,
               sycl::buffer<int32_t, 1> &co) {
    gemm_bias_precondition(selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda,
                           ao, b, ldb, bo, beta, c, ldc, co);
    oneapi::mkl::blas::mklgpu::MAJOR::gemm_bias(selector.get_queue(), transa, transb, offsetc, m, n,
                                                k, alpha, a, lda, ao, b, ldb, bo, beta, c, ldc, co);
    gemm_bias_postcondition(selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda,
                            ao, b, ldb, bo, beta, c, ldc, co);
}

void gemm_bias(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
               offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
               sycl::buffer<uint8_t, 1> &a, std::int64_t lda, uint8_t ao,
               sycl::buffer<int8_t, 1> &b, std::int64_t ldb, int8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, std::int64_t ldc,
               sycl::buffer<int32_t, 1> &co) {
    gemm_bias_precondition(selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda,
                           ao, b, ldb, bo, beta, c, ldc, co);
    oneapi::mkl::blas::mklgpu::MAJOR::gemm_bias(selector.get_queue(), transa, transb, offsetc, m, n,
                                                k, alpha, a, lda, ao, b, ldb, bo, beta, c, ldc, co);
    gemm_bias_postcondition(selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda,
                            ao, b, ldb, bo, beta, c, ldc, co);
}

void gemm_bias(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
               offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
               sycl::buffer<uint8_t, 1> &a, std::int64_t lda, uint8_t ao,
               sycl::buffer<uint8_t, 1> &b, std::int64_t ldb, uint8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, std::int64_t ldc,
               sycl::buffer<int32_t, 1> &co) {
    gemm_bias_precondition(selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda,
                           ao, b, ldb, bo, beta, c, ldc, co);
    oneapi::mkl::blas::mklgpu::MAJOR::gemm_bias(selector.get_queue(), transa, transb, offsetc, m, n,
                                                k, alpha, a, lda, ao, b, ldb, bo, beta, c, ldc, co);
    gemm_bias_postcondition(selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda,
                            ao, b, ldb, bo, beta, c, ldc, co);
}

void swap(backend_selector<backend::mklgpu> selector, std::int64_t n, sycl::buffer<float, 1> &x,
          std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy) {
    swap_precondition(selector.get_queue(), n, x, incx, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::swap(selector.get_queue(), n, x, incx, y, incy);
    swap_postcondition(selector.get_queue(), n, x, incx, y, incy);
}

void swap(backend_selector<backend::mklgpu> selector, std::int64_t n,
          sycl::buffer<double, 1> &x, std::int64_t incx, sycl::buffer<double, 1> &y,
          std::int64_t incy) {
    swap_precondition(selector.get_queue(), n, x, incx, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::swap(selector.get_queue(), n, x, incx, y, incy);
    swap_postcondition(selector.get_queue(), n, x, incx, y, incy);
}

void swap(backend_selector<backend::mklgpu> selector, std::int64_t n,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    swap_precondition(selector.get_queue(), n, x, incx, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::swap(selector.get_queue(), n, x, incx, y, incy);
    swap_postcondition(selector.get_queue(), n, x, incx, y, incy);
}

void swap(backend_selector<backend::mklgpu> selector, std::int64_t n,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    swap_precondition(selector.get_queue(), n, x, incx, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::swap(selector.get_queue(), n, x, incx, y, incy);
    swap_postcondition(selector.get_queue(), n, x, incx, y, incy);
}

void geru(backend_selector<backend::mklgpu> selector, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    geru_precondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::blas::mklgpu::MAJOR::geru(selector.get_queue(), m, n, alpha, x, incx, y, incy, a,
                                           lda);
    geru_postcondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda);
}

void geru(backend_selector<backend::mklgpu> selector, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    geru_precondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::blas::mklgpu::MAJOR::geru(selector.get_queue(), m, n, alpha, x, incx, y, incy, a,
                                           lda);
    geru_postcondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda);
}

void nrm2(backend_selector<backend::mklgpu> selector, std::int64_t n,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<float, 1> &result) {
    nrm2_precondition(selector.get_queue(), n, x, incx, result);
    oneapi::mkl::blas::mklgpu::MAJOR::nrm2(selector.get_queue(), n, x, incx, result);
    nrm2_postcondition(selector.get_queue(), n, x, incx, result);
}

void nrm2(backend_selector<backend::mklgpu> selector, std::int64_t n,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          sycl::buffer<double, 1> &result) {
    nrm2_precondition(selector.get_queue(), n, x, incx, result);
    oneapi::mkl::blas::mklgpu::MAJOR::nrm2(selector.get_queue(), n, x, incx, result);
    nrm2_postcondition(selector.get_queue(), n, x, incx, result);
}

void nrm2(backend_selector<backend::mklgpu> selector, std::int64_t n, sycl::buffer<float, 1> &x,
          std::int64_t incx, sycl::buffer<float, 1> &result) {
    nrm2_precondition(selector.get_queue(), n, x, incx, result);
    oneapi::mkl::blas::mklgpu::MAJOR::nrm2(selector.get_queue(), n, x, incx, result);
    nrm2_postcondition(selector.get_queue(), n, x, incx, result);
}

void nrm2(backend_selector<backend::mklgpu> selector, std::int64_t n,
          sycl::buffer<double, 1> &x, std::int64_t incx, sycl::buffer<double, 1> &result) {
    nrm2_precondition(selector.get_queue(), n, x, incx, result);
    oneapi::mkl::blas::mklgpu::MAJOR::nrm2(selector.get_queue(), n, x, incx, result);
    nrm2_postcondition(selector.get_queue(), n, x, incx, result);
}

void gemm(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
          sycl::buffer<float, 1> &a, std::int64_t lda, sycl::buffer<float, 1> &b,
          std::int64_t ldb, float beta, sycl::buffer<float, 1> &c, std::int64_t ldc) {
    gemm_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                      ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::gemm(selector.get_queue(), transa, transb, m, n, k, alpha, a,
                                           lda, b, ldb, beta, c, ldc);
    gemm_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc);
}

void gemm(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
          sycl::buffer<double, 1> &a, std::int64_t lda, sycl::buffer<double, 1> &b,
          std::int64_t ldb, double beta, sycl::buffer<double, 1> &c, std::int64_t ldc) {
    gemm_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                      ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::gemm(selector.get_queue(), transa, transb, m, n, k, alpha, a,
                                           lda, b, ldb, beta, c, ldc);
    gemm_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc);
}

void gemm(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    gemm_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                      ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::gemm(selector.get_queue(), transa, transb, m, n, k, alpha, a,
                                           lda, b, ldb, beta, c, ldc);
    gemm_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc);
}

void gemm(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    gemm_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                      ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::gemm(selector.get_queue(), transa, transb, m, n, k, alpha, a,
                                           lda, b, ldb, beta, c, ldc);
    gemm_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc);
}

void gemm(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, sycl::half alpha,
          sycl::buffer<sycl::half, 1> &a, std::int64_t lda, sycl::buffer<sycl::half, 1> &b,
          std::int64_t ldb, sycl::half beta, sycl::buffer<sycl::half, 1> &c, std::int64_t ldc) {
    gemm_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                      ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::gemm(selector.get_queue(), transa, transb, m, n, k, alpha, a,
                                           lda, b, ldb, beta, c, ldc);
    gemm_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc);
}

void gemm(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
          sycl::buffer<sycl::half, 1> &a, std::int64_t lda, sycl::buffer<sycl::half, 1> &b,
          std::int64_t ldb, float beta, sycl::buffer<float, 1> &c, std::int64_t ldc) {
    gemm_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                      ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::gemm(selector.get_queue(), transa, transb, m, n, k, alpha, a,
                                           lda, b, ldb, beta, c, ldc);
    gemm_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc);
}

void gemm(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
          sycl::buffer<bfloat16, 1> &a, std::int64_t lda, sycl::buffer<bfloat16, 1> &b,
          std::int64_t ldb, float beta, sycl::buffer<float, 1> &c, std::int64_t ldc) {
    gemm_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                      ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::gemm(selector.get_queue(), transa, transb, m, n, k, alpha, a,
                                           lda, b, ldb, beta, c, ldc);
    gemm_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc);
}

void syr2(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n, float alpha,
          sycl::buffer<float, 1> &x, std::int64_t incx, sycl::buffer<float, 1> &y,
          std::int64_t incy, sycl::buffer<float, 1> &a, std::int64_t lda) {
    syr2_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::blas::mklgpu::MAJOR::syr2(selector.get_queue(), upper_lower, n, alpha, x, incx, y,
                                           incy, a, lda);
    syr2_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

void syr2(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
          double alpha, sycl::buffer<double, 1> &x, std::int64_t incx,
          sycl::buffer<double, 1> &y, std::int64_t incy, sycl::buffer<double, 1> &a,
          std::int64_t lda) {
    syr2_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::blas::mklgpu::MAJOR::syr2(selector.get_queue(), upper_lower, n, alpha, x, incx, y,
                                           incy, a, lda);
    syr2_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

void ger(backend_selector<backend::mklgpu> selector, std::int64_t m, std::int64_t n, float alpha,
         sycl::buffer<float, 1> &x, std::int64_t incx, sycl::buffer<float, 1> &y,
         std::int64_t incy, sycl::buffer<float, 1> &a, std::int64_t lda) {
    ger_precondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::blas::mklgpu::MAJOR::ger(selector.get_queue(), m, n, alpha, x, incx, y, incy, a,
                                          lda);
    ger_postcondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda);
}

void ger(backend_selector<backend::mklgpu> selector, std::int64_t m, std::int64_t n, double alpha,
         sycl::buffer<double, 1> &x, std::int64_t incx, sycl::buffer<double, 1> &y,
         std::int64_t incy, sycl::buffer<double, 1> &a, std::int64_t lda) {
    ger_precondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::blas::mklgpu::MAJOR::ger(selector.get_queue(), m, n, alpha, x, incx, y, incy, a,
                                          lda);
    ger_postcondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda);
}

void trsm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
          transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
          sycl::buffer<float, 1> &a, std::int64_t lda, sycl::buffer<float, 1> &b,
          std::int64_t ldb) {
    trsm_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                      a, lda, b, ldb);
    oneapi::mkl::blas::mklgpu::MAJOR::trsm(selector.get_queue(), left_right, upper_lower, trans,
                                           unit_diag, m, n, alpha, a, lda, b, ldb);
    trsm_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                       a, lda, b, ldb);
}

void trsm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
          transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
          sycl::buffer<double, 1> &a, std::int64_t lda, sycl::buffer<double, 1> &b,
          std::int64_t ldb) {
    trsm_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                      a, lda, b, ldb);
    oneapi::mkl::blas::mklgpu::MAJOR::trsm(selector.get_queue(), left_right, upper_lower, trans,
                                           unit_diag, m, n, alpha, a, lda, b, ldb);
    trsm_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                       a, lda, b, ldb);
}

void trsm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
          transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    trsm_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                      a, lda, b, ldb);
    oneapi::mkl::blas::mklgpu::MAJOR::trsm(selector.get_queue(), left_right, upper_lower, trans,
                                           unit_diag, m, n, alpha, a, lda, b, ldb);
    trsm_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                       a, lda, b, ldb);
}

void trsm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
          transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    trsm_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                      a, lda, b, ldb);
    oneapi::mkl::blas::mklgpu::MAJOR::trsm(selector.get_queue(), left_right, upper_lower, trans,
                                           unit_diag, m, n, alpha, a, lda, b, ldb);
    trsm_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                       a, lda, b, ldb);
}

void dotu(backend_selector<backend::mklgpu> selector, std::int64_t n,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1> &result) {
    dotu_precondition(selector.get_queue(), n, x, incx, y, incy, result);
    oneapi::mkl::blas::mklgpu::MAJOR::dotu(selector.get_queue(), n, x, incx, y, incy, result);
    dotu_postcondition(selector.get_queue(), n, x, incx, y, incy, result);
}

void dotu(backend_selector<backend::mklgpu> selector, std::int64_t n,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1> &result) {
    dotu_precondition(selector.get_queue(), n, x, incx, y, incy, result);
    oneapi::mkl::blas::mklgpu::MAJOR::dotu(selector.get_queue(), n, x, incx, y, incy, result);
    dotu_postcondition(selector.get_queue(), n, x, incx, y, incy, result);
}

void hemm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    hemm_precondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                      beta, c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::hemm(selector.get_queue(), left_right, upper_lower, m, n,
                                           alpha, a, lda, b, ldb, beta, c, ldc);
    hemm_postcondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                       beta, c, ldc);
}

void hemm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    hemm_precondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                      beta, c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::hemm(selector.get_queue(), left_right, upper_lower, m, n,
                                           alpha, a, lda, b, ldb, beta, c, ldc);
    hemm_postcondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                       beta, c, ldc);
}

void hpr2(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1> &a) {
    hpr2_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a);
    oneapi::mkl::blas::mklgpu::MAJOR::hpr2(selector.get_queue(), upper_lower, n, alpha, x, incx, y,
                                           incy, a);
    hpr2_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a);
}

void hpr2(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1> &a) {
    hpr2_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a);
    oneapi::mkl::blas::mklgpu::MAJOR::hpr2(selector.get_queue(), upper_lower, n, alpha, x, incx, y,
                                           incy, a);
    hpr2_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a);
}

void gbmv(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
          std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha,
          sycl::buffer<float, 1> &a, std::int64_t lda, sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, sycl::buffer<float, 1> &y, std::int64_t incy) {
    gbmv_precondition(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                      incy);
    oneapi::mkl::blas::mklgpu::MAJOR::gbmv(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda,
                                           x, incx, beta, y, incy);
    gbmv_postcondition(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                       incy);
}

void gbmv(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
          std::int64_t n, std::int64_t kl, std::int64_t ku, double alpha,
          sycl::buffer<double, 1> &a, std::int64_t lda, sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, sycl::buffer<double, 1> &y, std::int64_t incy) {
    gbmv_precondition(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                      incy);
    oneapi::mkl::blas::mklgpu::MAJOR::gbmv(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda,
                                           x, incx, beta, y, incy);
    gbmv_postcondition(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                       incy);
}

void gbmv(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
          std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    gbmv_precondition(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                      incy);
    oneapi::mkl::blas::mklgpu::MAJOR::gbmv(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda,
                                           x, incx, beta, y, incy);
    gbmv_postcondition(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                       incy);
}

void gbmv(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
          std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    gbmv_precondition(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                      incy);
    oneapi::mkl::blas::mklgpu::MAJOR::gbmv(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda,
                                           x, incx, beta, y, incy);
    gbmv_postcondition(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                       incy);
}

void tbmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, std::int64_t k, sycl::buffer<float, 1> &a,
          std::int64_t lda, sycl::buffer<float, 1> &x, std::int64_t incx) {
    tbmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::tbmv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           k, a, lda, x, incx);
    tbmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, std::int64_t k, sycl::buffer<double, 1> &a,
          std::int64_t lda, sycl::buffer<double, 1> &x, std::int64_t incx) {
    tbmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::tbmv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           k, a, lda, x, incx);
    tbmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, std::int64_t k,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    tbmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::tbmv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           k, a, lda, x, incx);
    tbmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, std::int64_t k,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    tbmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::tbmv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           k, a, lda, x, incx);
    tbmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void symm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
          std::int64_t m, std::int64_t n, float alpha, sycl::buffer<float, 1> &a,
          std::int64_t lda, sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
          sycl::buffer<float, 1> &c, std::int64_t ldc) {
    symm_precondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                      beta, c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::symm(selector.get_queue(), left_right, upper_lower, m, n,
                                           alpha, a, lda, b, ldb, beta, c, ldc);
    symm_postcondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                       beta, c, ldc);
}

void symm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
          std::int64_t m, std::int64_t n, double alpha, sycl::buffer<double, 1> &a,
          std::int64_t lda, sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
          sycl::buffer<double, 1> &c, std::int64_t ldc) {
    symm_precondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                      beta, c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::symm(selector.get_queue(), left_right, upper_lower, m, n,
                                           alpha, a, lda, b, ldb, beta, c, ldc);
    symm_postcondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                       beta, c, ldc);
}

void symm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    symm_precondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                      beta, c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::symm(selector.get_queue(), left_right, upper_lower, m, n,
                                           alpha, a, lda, b, ldb, beta, c, ldc);
    symm_postcondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                       beta, c, ldc);
}

void symm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    symm_precondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                      beta, c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::symm(selector.get_queue(), left_right, upper_lower, m, n,
                                           alpha, a, lda, b, ldb, beta, c, ldc);
    symm_postcondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                       beta, c, ldc);
}

void dotc(backend_selector<backend::mklgpu> selector, std::int64_t n,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1> &result) {
    dotc_precondition(selector.get_queue(), n, x, incx, y, incy, result);
    oneapi::mkl::blas::mklgpu::MAJOR::dotc(selector.get_queue(), n, x, incx, y, incy, result);
    dotc_postcondition(selector.get_queue(), n, x, incx, y, incy, result);
}

void dotc(backend_selector<backend::mklgpu> selector, std::int64_t n,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1> &result) {
    dotc_precondition(selector.get_queue(), n, x, incx, y, incy, result);
    oneapi::mkl::blas::mklgpu::MAJOR::dotc(selector.get_queue(), n, x, incx, y, incy, result);
    dotc_postcondition(selector.get_queue(), n, x, incx, y, incy, result);
}

void syr(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n, float alpha,
         sycl::buffer<float, 1> &x, std::int64_t incx, sycl::buffer<float, 1> &a,
         std::int64_t lda) {
    syr_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, lda);
    oneapi::mkl::blas::mklgpu::MAJOR::syr(selector.get_queue(), upper_lower, n, alpha, x, incx, a,
                                          lda);
    syr_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, lda);
}

void syr(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n, double alpha,
         sycl::buffer<double, 1> &x, std::int64_t incx, sycl::buffer<double, 1> &a,
         std::int64_t lda) {
    syr_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, lda);
    oneapi::mkl::blas::mklgpu::MAJOR::syr(selector.get_queue(), upper_lower, n, alpha, x, incx, a,
                                          lda);
    syr_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, lda);
}

void trmm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
          transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
          sycl::buffer<float, 1> &a, std::int64_t lda, sycl::buffer<float, 1> &b,
          std::int64_t ldb) {
    trmm_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                      a, lda, b, ldb);
    oneapi::mkl::blas::mklgpu::MAJOR::trmm(selector.get_queue(), left_right, upper_lower, trans,
                                           unit_diag, m, n, alpha, a, lda, b, ldb);
    trmm_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                       a, lda, b, ldb);
}

void trmm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
          transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
          sycl::buffer<double, 1> &a, std::int64_t lda, sycl::buffer<double, 1> &b,
          std::int64_t ldb) {
    trmm_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                      a, lda, b, ldb);
    oneapi::mkl::blas::mklgpu::MAJOR::trmm(selector.get_queue(), left_right, upper_lower, trans,
                                           unit_diag, m, n, alpha, a, lda, b, ldb);
    trmm_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                       a, lda, b, ldb);
}

void trmm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
          transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    trmm_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                      a, lda, b, ldb);
    oneapi::mkl::blas::mklgpu::MAJOR::trmm(selector.get_queue(), left_right, upper_lower, trans,
                                           unit_diag, m, n, alpha, a, lda, b, ldb);
    trmm_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                       a, lda, b, ldb);
}

void trmm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
          transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    trmm_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                      a, lda, b, ldb);
    oneapi::mkl::blas::mklgpu::MAJOR::trmm(selector.get_queue(), left_right, upper_lower, trans,
                                           unit_diag, m, n, alpha, a, lda, b, ldb);
    trmm_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                       a, lda, b, ldb);
}

void rotmg(backend_selector<backend::mklgpu> selector, sycl::buffer<float, 1> &d1,
           sycl::buffer<float, 1> &d2, sycl::buffer<float, 1> &x1, float y1,
           sycl::buffer<float, 1> &param) {
    rotmg_precondition(selector.get_queue(), d1, d2, x1, y1, param);
    oneapi::mkl::blas::mklgpu::MAJOR::rotmg(selector.get_queue(), d1, d2, x1, y1, param);
    rotmg_postcondition(selector.get_queue(), d1, d2, x1, y1, param);
}

void rotmg(backend_selector<backend::mklgpu> selector, sycl::buffer<double, 1> &d1,
           sycl::buffer<double, 1> &d2, sycl::buffer<double, 1> &x1, double y1,
           sycl::buffer<double, 1> &param) {
    rotmg_precondition(selector.get_queue(), d1, d2, x1, y1, param);
    oneapi::mkl::blas::mklgpu::MAJOR::rotmg(selector.get_queue(), d1, d2, x1, y1, param);
    rotmg_postcondition(selector.get_queue(), d1, d2, x1, y1, param);
}

void tpsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, sycl::buffer<float, 1> &a,
          sycl::buffer<float, 1> &x, std::int64_t incx) {
    tpsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::tpsv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           a, x, incx);
    tpsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, sycl::buffer<double, 1> &a,
          sycl::buffer<double, 1> &x, std::int64_t incx) {
    tpsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::tpsv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           a, x, incx);
    tpsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, sycl::buffer<std::complex<float>, 1> &a,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    tpsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::tpsv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           a, x, incx);
    tpsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, sycl::buffer<std::complex<double>, 1> &a,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    tpsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::tpsv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           a, x, incx);
    tpsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx);
}

void trsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, sycl::buffer<float, 1> &a, std::int64_t lda,
          sycl::buffer<float, 1> &x, std::int64_t incx) {
    trsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::trsv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           a, lda, x, incx);
    trsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, sycl::buffer<double, 1> &a, std::int64_t lda,
          sycl::buffer<double, 1> &x, std::int64_t incx) {
    trsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::trsv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           a, lda, x, incx);
    trsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    trsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::trsv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           a, lda, x, incx);
    trsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    trsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::trsv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           a, lda, x, incx);
    trsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void copy(backend_selector<backend::mklgpu> selector, std::int64_t n, sycl::buffer<float, 1> &x,
          std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy) {
    copy_precondition(selector.get_queue(), n, x, incx, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::copy(selector.get_queue(), n, x, incx, y, incy);
    copy_postcondition(selector.get_queue(), n, x, incx, y, incy);
}

void copy(backend_selector<backend::mklgpu> selector, std::int64_t n,
          sycl::buffer<double, 1> &x, std::int64_t incx, sycl::buffer<double, 1> &y,
          std::int64_t incy) {
    copy_precondition(selector.get_queue(), n, x, incx, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::copy(selector.get_queue(), n, x, incx, y, incy);
    copy_postcondition(selector.get_queue(), n, x, incx, y, incy);
}

void copy(backend_selector<backend::mklgpu> selector, std::int64_t n,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    copy_precondition(selector.get_queue(), n, x, incx, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::copy(selector.get_queue(), n, x, incx, y, incy);
    copy_postcondition(selector.get_queue(), n, x, incx, y, incy);
}

void copy(backend_selector<backend::mklgpu> selector, std::int64_t n,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    copy_precondition(selector.get_queue(), n, x, incx, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::copy(selector.get_queue(), n, x, incx, y, incy);
    copy_postcondition(selector.get_queue(), n, x, incx, y, incy);
}

void copy_batch(backend_selector<backend::mklgpu> selector, std::int64_t n,
                sycl::buffer<float, 1> &x, std::int64_t incx, std::int64_t stridex,
                sycl::buffer<float, 1> &y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size) {
    copy_batch_precondition(selector.get_queue(), n, x, incx, stridex, y, incy, stridey,
                            batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::copy_batch(selector.get_queue(), n, x, incx, stridex, y, incy,
                                                 stridey, batch_size);
    copy_batch_postcondition(selector.get_queue(), n, x, incx, stridex, y, incy, stridey,
                             batch_size);
}

void copy_batch(backend_selector<backend::mklgpu> selector, std::int64_t n,
                sycl::buffer<double, 1> &x, std::int64_t incx, std::int64_t stridex,
                sycl::buffer<double, 1> &y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size) {
    copy_batch_precondition(selector.get_queue(), n, x, incx, stridex, y, incy, stridey,
                            batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::copy_batch(selector.get_queue(), n, x, incx, stridex, y, incy,
                                                 stridey, batch_size);
    copy_batch_postcondition(selector.get_queue(), n, x, incx, stridex, y, incy, stridey,
                             batch_size);
}

void copy_batch(backend_selector<backend::mklgpu> selector, std::int64_t n,
                sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                std::int64_t stridex, sycl::buffer<std::complex<float>, 1> &y,
                std::int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
    copy_batch_precondition(selector.get_queue(), n, x, incx, stridex, y, incy, stridey,
                            batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::copy_batch(selector.get_queue(), n, x, incx, stridex, y, incy,
                                                 stridey, batch_size);
    copy_batch_postcondition(selector.get_queue(), n, x, incx, stridex, y, incy, stridey,
                             batch_size);
}

void copy_batch(backend_selector<backend::mklgpu> selector, std::int64_t n,
                sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                std::int64_t stridex, sycl::buffer<std::complex<double>, 1> &y,
                std::int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
    copy_batch_precondition(selector.get_queue(), n, x, incx, stridex, y, incy, stridey,
                            batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::copy_batch(selector.get_queue(), n, x, incx, stridex, y, incy,
                                                 stridey, batch_size);
    copy_batch_postcondition(selector.get_queue(), n, x, incx, stridex, y, incy, stridey,
                             batch_size);
}

void hemv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    hemv_precondition(selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::hemv(selector.get_queue(), upper_lower, n, alpha, a, lda, x,
                                           incx, beta, y, incy);
    hemv_postcondition(selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

void hemv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    hemv_precondition(selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::hemv(selector.get_queue(), upper_lower, n, alpha, a, lda, x,
                                           incx, beta, y, incy);
    hemv_postcondition(selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemmt(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose transa,
           transpose transb, std::int64_t n, std::int64_t k, float alpha,
           sycl::buffer<float, 1> &a, std::int64_t lda, sycl::buffer<float, 1> &b,
           std::int64_t ldb, float beta, sycl::buffer<float, 1> &c, std::int64_t ldc) {
    gemmt_precondition(selector.get_queue(), upper_lower, transa, transb, n, k, alpha, a, lda, b,
                       ldb, beta, c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::gemmt(selector.get_queue(), upper_lower, transa, transb, n, k,
                                            alpha, a, lda, b, ldb, beta, c, ldc);
    gemmt_postcondition(selector.get_queue(), upper_lower, transa, transb, n, k, alpha, a, lda, b,
                        ldb, beta, c, ldc);
}

void gemmt(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose transa,
           transpose transb, std::int64_t n, std::int64_t k, double alpha,
           sycl::buffer<double, 1> &a, std::int64_t lda, sycl::buffer<double, 1> &b,
           std::int64_t ldb, double beta, sycl::buffer<double, 1> &c, std::int64_t ldc) {
    gemmt_precondition(selector.get_queue(), upper_lower, transa, transb, n, k, alpha, a, lda, b,
                       ldb, beta, c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::gemmt(selector.get_queue(), upper_lower, transa, transb, n, k,
                                            alpha, a, lda, b, ldb, beta, c, ldc);
    gemmt_postcondition(selector.get_queue(), upper_lower, transa, transb, n, k, alpha, a, lda, b,
                        ldb, beta, c, ldc);
}

void gemmt(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose transa,
           transpose transb, std::int64_t n, std::int64_t k, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    gemmt_precondition(selector.get_queue(), upper_lower, transa, transb, n, k, alpha, a, lda, b,
                       ldb, beta, c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::gemmt(selector.get_queue(), upper_lower, transa, transb, n, k,
                                            alpha, a, lda, b, ldb, beta, c, ldc);
    gemmt_postcondition(selector.get_queue(), upper_lower, transa, transb, n, k, alpha, a, lda, b,
                        ldb, beta, c, ldc);
}

void gemmt(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose transa,
           transpose transb, std::int64_t n, std::int64_t k, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc) {
    gemmt_precondition(selector.get_queue(), upper_lower, transa, transb, n, k, alpha, a, lda, b,
                       ldb, beta, c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::gemmt(selector.get_queue(), upper_lower, transa, transb, n, k,
                                            alpha, a, lda, b, ldb, beta, c, ldc);
    gemmt_postcondition(selector.get_queue(), upper_lower, transa, transb, n, k, alpha, a, lda, b,
                        ldb, beta, c, ldc);
}

void asum(backend_selector<backend::mklgpu> selector, std::int64_t n,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<float, 1> &result) {
    asum_precondition(selector.get_queue(), n, x, incx, result);
    oneapi::mkl::blas::mklgpu::MAJOR::asum(selector.get_queue(), n, x, incx, result);
    asum_postcondition(selector.get_queue(), n, x, incx, result);
}

void asum(backend_selector<backend::mklgpu> selector, std::int64_t n,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          sycl::buffer<double, 1> &result) {
    asum_precondition(selector.get_queue(), n, x, incx, result);
    oneapi::mkl::blas::mklgpu::MAJOR::asum(selector.get_queue(), n, x, incx, result);
    asum_postcondition(selector.get_queue(), n, x, incx, result);
}

void asum(backend_selector<backend::mklgpu> selector, std::int64_t n, sycl::buffer<float, 1> &x,
          std::int64_t incx, sycl::buffer<float, 1> &result) {
    asum_precondition(selector.get_queue(), n, x, incx, result);
    oneapi::mkl::blas::mklgpu::MAJOR::asum(selector.get_queue(), n, x, incx, result);
    asum_postcondition(selector.get_queue(), n, x, incx, result);
}

void asum(backend_selector<backend::mklgpu> selector, std::int64_t n,
          sycl::buffer<double, 1> &x, std::int64_t incx, sycl::buffer<double, 1> &result) {
    asum_precondition(selector.get_queue(), n, x, incx, result);
    oneapi::mkl::blas::mklgpu::MAJOR::asum(selector.get_queue(), n, x, incx, result);
    asum_postcondition(selector.get_queue(), n, x, incx, result);
}

void sbmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
          std::int64_t k, float alpha, sycl::buffer<float, 1> &a, std::int64_t lda,
          sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          sycl::buffer<float, 1> &y, std::int64_t incy) {
    sbmv_precondition(selector.get_queue(), upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                      incy);
    oneapi::mkl::blas::mklgpu::MAJOR::sbmv(selector.get_queue(), upper_lower, n, k, alpha, a, lda,
                                           x, incx, beta, y, incy);
    sbmv_postcondition(selector.get_queue(), upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                       incy);
}

void sbmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
          std::int64_t k, double alpha, sycl::buffer<double, 1> &a, std::int64_t lda,
          sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          sycl::buffer<double, 1> &y, std::int64_t incy) {
    sbmv_precondition(selector.get_queue(), upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                      incy);
    oneapi::mkl::blas::mklgpu::MAJOR::sbmv(selector.get_queue(), upper_lower, n, k, alpha, a, lda,
                                           x, incx, beta, y, incy);
    sbmv_postcondition(selector.get_queue(), upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                       incy);
}

void tbsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, std::int64_t k, sycl::buffer<float, 1> &a,
          std::int64_t lda, sycl::buffer<float, 1> &x, std::int64_t incx) {
    tbsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::tbsv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           k, a, lda, x, incx);
    tbsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, std::int64_t k, sycl::buffer<double, 1> &a,
          std::int64_t lda, sycl::buffer<double, 1> &x, std::int64_t incx) {
    tbsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::tbsv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           k, a, lda, x, incx);
    tbsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, std::int64_t k,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    tbsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::tbsv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           k, a, lda, x, incx);
    tbsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, std::int64_t k,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    tbsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    oneapi::mkl::blas::mklgpu::MAJOR::tbsv(selector.get_queue(), upper_lower, trans, unit_diag, n,
                                           k, a, lda, x, incx);
    tbsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void spr2(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n, float alpha,
          sycl::buffer<float, 1> &x, std::int64_t incx, sycl::buffer<float, 1> &y,
          std::int64_t incy, sycl::buffer<float, 1> &a) {
    spr2_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a);
    oneapi::mkl::blas::mklgpu::MAJOR::spr2(selector.get_queue(), upper_lower, n, alpha, x, incx, y,
                                           incy, a);
    spr2_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a);
}

void spr2(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
          double alpha, sycl::buffer<double, 1> &x, std::int64_t incx,
          sycl::buffer<double, 1> &y, std::int64_t incy, sycl::buffer<double, 1> &a) {
    spr2_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a);
    oneapi::mkl::blas::mklgpu::MAJOR::spr2(selector.get_queue(), upper_lower, n, alpha, x, incx, y,
                                           incy, a);
    spr2_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a);
}

void iamax(backend_selector<backend::mklgpu> selector, std::int64_t n,
           sycl::buffer<float, 1> &x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(selector.get_queue(), n, x, incx, result);
    oneapi::mkl::blas::mklgpu::MAJOR::iamax(selector.get_queue(), n, x, incx, result);
    iamax_postcondition(selector.get_queue(), n, x, incx, result);
}

void iamax(backend_selector<backend::mklgpu> selector, std::int64_t n,
           sycl::buffer<double, 1> &x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(selector.get_queue(), n, x, incx, result);
    oneapi::mkl::blas::mklgpu::MAJOR::iamax(selector.get_queue(), n, x, incx, result);
    iamax_postcondition(selector.get_queue(), n, x, incx, result);
}

void iamax(backend_selector<backend::mklgpu> selector, std::int64_t n,
           sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(selector.get_queue(), n, x, incx, result);
    oneapi::mkl::blas::mklgpu::MAJOR::iamax(selector.get_queue(), n, x, incx, result);
    iamax_postcondition(selector.get_queue(), n, x, incx, result);
}

void iamax(backend_selector<backend::mklgpu> selector, std::int64_t n,
           sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(selector.get_queue(), n, x, incx, result);
    oneapi::mkl::blas::mklgpu::MAJOR::iamax(selector.get_queue(), n, x, incx, result);
    iamax_postcondition(selector.get_queue(), n, x, incx, result);
}

void rotm(backend_selector<backend::mklgpu> selector, std::int64_t n, sycl::buffer<float, 1> &x,
          std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy,
          sycl::buffer<float, 1> &param) {
    rotm_precondition(selector.get_queue(), n, x, incx, y, incy, param);
    oneapi::mkl::blas::mklgpu::MAJOR::rotm(selector.get_queue(), n, x, incx, y, incy, param);
    rotm_postcondition(selector.get_queue(), n, x, incx, y, incy, param);
}

void rotm(backend_selector<backend::mklgpu> selector, std::int64_t n,
          sycl::buffer<double, 1> &x, std::int64_t incx, sycl::buffer<double, 1> &y,
          std::int64_t incy, sycl::buffer<double, 1> &param) {
    rotm_precondition(selector.get_queue(), n, x, incx, y, incy, param);
    oneapi::mkl::blas::mklgpu::MAJOR::rotm(selector.get_queue(), n, x, incx, y, incy, param);
    rotm_postcondition(selector.get_queue(), n, x, incx, y, incy, param);
}

void dot(backend_selector<backend::mklgpu> selector, std::int64_t n, sycl::buffer<float, 1> &x,
         std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy,
         sycl::buffer<float, 1> &result) {
    dot_precondition(selector.get_queue(), n, x, incx, y, incy, result);
    oneapi::mkl::blas::mklgpu::MAJOR::dot(selector.get_queue(), n, x, incx, y, incy, result);
    dot_postcondition(selector.get_queue(), n, x, incx, y, incy, result);
}

void dot(backend_selector<backend::mklgpu> selector, std::int64_t n, sycl::buffer<double, 1> &x,
         std::int64_t incx, sycl::buffer<double, 1> &y, std::int64_t incy,
         sycl::buffer<double, 1> &result) {
    dot_precondition(selector.get_queue(), n, x, incx, y, incy, result);
    oneapi::mkl::blas::mklgpu::MAJOR::dot(selector.get_queue(), n, x, incx, y, incy, result);
    dot_postcondition(selector.get_queue(), n, x, incx, y, incy, result);
}

void dot(backend_selector<backend::mklgpu> selector, std::int64_t n, sycl::buffer<float, 1> &x,
         std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy,
         sycl::buffer<double, 1> &result) {
    dot_precondition(selector.get_queue(), n, x, incx, y, incy, result);
    oneapi::mkl::blas::mklgpu::MAJOR::dot(selector.get_queue(), n, x, incx, y, incy, result);
    dot_postcondition(selector.get_queue(), n, x, incx, y, incy, result);
}

void trsm_batch(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
                transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size) {
    trsm_batch_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                            alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::trsm_batch(selector.get_queue(), left_right, upper_lower,
                                                 trans, unit_diag, m, n, alpha, a, lda, stride_a, b,
                                                 ldb, stride_b, batch_size);
    trsm_batch_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                             alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
}

void trsm_batch(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
                transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size) {
    trsm_batch_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                            alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::trsm_batch(selector.get_queue(), left_right, upper_lower,
                                                 trans, unit_diag, m, n, alpha, a, lda, stride_a, b,
                                                 ldb, stride_b, batch_size);
    trsm_batch_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                             alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
}

void trsm_batch(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
                transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    trsm_batch_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                            alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::trsm_batch(selector.get_queue(), left_right, upper_lower,
                                                 trans, unit_diag, m, n, alpha, a, lda, stride_a, b,
                                                 ldb, stride_b, batch_size);
    trsm_batch_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                             alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
}

void trsm_batch(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
                transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    trsm_batch_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                            alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::trsm_batch(selector.get_queue(), left_right, upper_lower,
                                                 trans, unit_diag, m, n, alpha, a, lda, stride_a, b,
                                                 ldb, stride_b, batch_size);
    trsm_batch_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                             alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
}

void her2k(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
           std::int64_t n, std::int64_t k, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, float beta,
           sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    her2k_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::her2k(selector.get_queue(), upper_lower, trans, n, k, alpha,
                                            a, lda, b, ldb, beta, c, ldc);
    her2k_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                        c, ldc);
}

void her2k(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
           std::int64_t n, std::int64_t k, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, double beta,
           sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    her2k_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc);
    oneapi::mkl::blas::mklgpu::MAJOR::her2k(selector.get_queue(), upper_lower, trans, n, k, alpha,
                                            a, lda, b, ldb, beta, c, ldc);
    her2k_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                        c, ldc);
}

void rotg(backend_selector<backend::mklgpu> selector, sycl::buffer<float, 1> &a,
          sycl::buffer<float, 1> &b, sycl::buffer<float, 1> &c,
          sycl::buffer<float, 1> &s) {
    rotg_precondition(selector.get_queue(), a, b, c, s);
    oneapi::mkl::blas::mklgpu::MAJOR::rotg(selector.get_queue(), a, b, c, s);
    rotg_postcondition(selector.get_queue(), a, b, c, s);
}

void rotg(backend_selector<backend::mklgpu> selector, sycl::buffer<double, 1> &a,
          sycl::buffer<double, 1> &b, sycl::buffer<double, 1> &c,
          sycl::buffer<double, 1> &s) {
    rotg_precondition(selector.get_queue(), a, b, c, s);
    oneapi::mkl::blas::mklgpu::MAJOR::rotg(selector.get_queue(), a, b, c, s);
    rotg_postcondition(selector.get_queue(), a, b, c, s);
}

void rotg(backend_selector<backend::mklgpu> selector, sycl::buffer<std::complex<float>, 1> &a,
          sycl::buffer<std::complex<float>, 1> &b, sycl::buffer<float, 1> &c,
          sycl::buffer<std::complex<float>, 1> &s) {
    rotg_precondition(selector.get_queue(), a, b, c, s);
    oneapi::mkl::blas::mklgpu::MAJOR::rotg(selector.get_queue(), a, b, c, s);
    rotg_postcondition(selector.get_queue(), a, b, c, s);
}

void rotg(backend_selector<backend::mklgpu> selector, sycl::buffer<std::complex<double>, 1> &a,
          sycl::buffer<std::complex<double>, 1> &b, sycl::buffer<double, 1> &c,
          sycl::buffer<std::complex<double>, 1> &s) {
    rotg_precondition(selector.get_queue(), a, b, c, s);
    oneapi::mkl::blas::mklgpu::MAJOR::rotg(selector.get_queue(), a, b, c, s);
    rotg_postcondition(selector.get_queue(), a, b, c, s);
}

void symv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n, float alpha,
          sycl::buffer<float, 1> &a, std::int64_t lda, sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, sycl::buffer<float, 1> &y, std::int64_t incy) {
    symv_precondition(selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::symv(selector.get_queue(), upper_lower, n, alpha, a, lda, x,
                                           incx, beta, y, incy);
    symv_postcondition(selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

void symv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
          double alpha, sycl::buffer<double, 1> &a, std::int64_t lda,
          sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          sycl::buffer<double, 1> &y, std::int64_t incy) {
    symv_precondition(selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::blas::mklgpu::MAJOR::symv(selector.get_queue(), upper_lower, n, alpha, a, lda, x,
                                           incx, beta, y, incy);
    symv_postcondition(selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

void omatcopy_batch(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                    std::int64_t n, float alpha, sycl::buffer<float, 1> &a, std::int64_t lda,
                    std::int64_t stride_a, sycl::buffer<float, 1> &b, std::int64_t ldb,
                    std::int64_t stride_b, std::int64_t batch_size) {
    omatcopy_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb,
                                stride_b, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::omatcopy_batch(selector.get_queue(), trans, m, n, alpha, a,
                                                     lda, stride_a, b, ldb, stride_b, batch_size);
    omatcopy_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb,
                                 stride_b, batch_size);
}

void omatcopy_batch(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                    std::int64_t n, double alpha, sycl::buffer<double, 1> &a, std::int64_t lda,
                    std::int64_t stride_a, sycl::buffer<double, 1> &b, std::int64_t ldb,
                    std::int64_t stride_b, std::int64_t batch_size) {
    omatcopy_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb,
                                stride_b, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::omatcopy_batch(selector.get_queue(), trans, m, n, alpha, a,
                                                     lda, stride_a, b, ldb, stride_b, batch_size);
    omatcopy_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb,
                                 stride_b, batch_size);
}

void omatcopy_batch(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                    std::int64_t n, std::complex<float> alpha,
                    sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                    std::int64_t stride_a, sycl::buffer<std::complex<float>, 1> &b,
                    std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    omatcopy_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb,
                                stride_b, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::omatcopy_batch(selector.get_queue(), trans, m, n, alpha, a,
                                                     lda, stride_a, b, ldb, stride_b, batch_size);
    omatcopy_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb,
                                 stride_b, batch_size);
}

void omatcopy_batch(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                    std::int64_t n, std::complex<double> alpha,
                    sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                    std::int64_t stride_a, sycl::buffer<std::complex<double>, 1> &b,
                    std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    omatcopy_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb,
                                stride_b, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::omatcopy_batch(selector.get_queue(), trans, m, n, alpha, a,
                                                     lda, stride_a, b, ldb, stride_b, batch_size);
    omatcopy_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb,
                                 stride_b, batch_size);
}

void imatcopy_batch(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                    std::int64_t n, float alpha, sycl::buffer<float, 1> &ab, std::int64_t lda,
                    std::int64_t ldb, std::int64_t stride, std::int64_t batch_size) {
    imatcopy_batch_precondition(selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride,
                                batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::imatcopy_batch(selector.get_queue(), trans, m, n, alpha, ab,
                                                     lda, ldb, stride, batch_size);
    imatcopy_batch_postcondition(selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride,
                                 batch_size);
}

void imatcopy_batch(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                    std::int64_t n, double alpha, sycl::buffer<double, 1> &ab, std::int64_t lda,
                    std::int64_t ldb, std::int64_t stride, std::int64_t batch_size) {
    imatcopy_batch_precondition(selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride,
                                batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::imatcopy_batch(selector.get_queue(), trans, m, n, alpha, ab,
                                                     lda, ldb, stride, batch_size);
    imatcopy_batch_postcondition(selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride,
                                 batch_size);
}

void imatcopy_batch(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                    std::int64_t n, std::complex<float> alpha,
                    sycl::buffer<std::complex<float>, 1> &ab, std::int64_t lda, std::int64_t ldb,
                    std::int64_t stride, std::int64_t batch_size) {
    imatcopy_batch_precondition(selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride,
                                batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::imatcopy_batch(selector.get_queue(), trans, m, n, alpha, ab,
                                                     lda, ldb, stride, batch_size);
    imatcopy_batch_postcondition(selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride,
                                 batch_size);
}

void imatcopy_batch(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                    std::int64_t n, std::complex<double> alpha,
                    sycl::buffer<std::complex<double>, 1> &ab, std::int64_t lda, std::int64_t ldb,
                    std::int64_t stride, std::int64_t batch_size) {
    imatcopy_batch_precondition(selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride,
                                batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::imatcopy_batch(selector.get_queue(), trans, m, n, alpha, ab,
                                                     lda, ldb, stride, batch_size);
    imatcopy_batch_postcondition(selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride,
                                 batch_size);
}

void omatadd_batch(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
                   std::int64_t m, std::int64_t n, float alpha, sycl::buffer<float, 1> &a,
                   std::int64_t lda, std::int64_t stride_a, float beta, sycl::buffer<float, 1> &b,
                   std::int64_t ldb, std::int64_t stride_b, sycl::buffer<float, 1> &c,
                   std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
    omatadd_batch_precondition(selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a,
                               beta, b, ldb, stride_b, c, ldc, stride_c, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::omatadd_batch(selector.get_queue(), transa, transb, m, n,
                                                    alpha, a, lda, stride_a, beta, b, ldb, stride_b,
                                                    c, ldc, stride_c, batch_size);
    omatadd_batch_postcondition(selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a,
                                beta, b, ldb, stride_b, c, ldc, stride_c, batch_size);
}

void omatadd_batch(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
                   std::int64_t m, std::int64_t n, double alpha, sycl::buffer<double, 1> &a,
                   std::int64_t lda, std::int64_t stride_a, double beta, sycl::buffer<double, 1> &b,
                   std::int64_t ldb, std::int64_t stride_b, sycl::buffer<double, 1> &c,
                   std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
    omatadd_batch_precondition(selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a,
                               beta, b, ldb, stride_b, c, ldc, stride_c, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::omatadd_batch(selector.get_queue(), transa, transb, m, n,
                                                    alpha, a, lda, stride_a, beta, b, ldb, stride_b,
                                                    c, ldc, stride_c, batch_size);
    omatadd_batch_postcondition(selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a,
                                beta, b, ldb, stride_b, c, ldc, stride_c, batch_size);
}

void omatadd_batch(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
                   std::int64_t m, std::int64_t n, std::complex<float> alpha,
                   sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda, std::int64_t stride_a,
                   std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &b,
                   std::int64_t ldb, std::int64_t stride_b, sycl::buffer<std::complex<float>, 1> &c,
                   std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
    omatadd_batch_precondition(selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a,
                               beta, b, ldb, stride_b, c, ldc, stride_c, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::omatadd_batch(selector.get_queue(), transa, transb, m, n,
                                                    alpha, a, lda, stride_a, beta, b, ldb, stride_b,
                                                    c, ldc, stride_c, batch_size);
    omatadd_batch_postcondition(selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a,
                                beta, b, ldb, stride_b, c, ldc, stride_c, batch_size);
}

void omatadd_batch(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
                   std::int64_t m, std::int64_t n, std::complex<double> alpha,
                   sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                   std::int64_t stride_a, std::complex<double> beta,
                   sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                   std::int64_t stride_b, sycl::buffer<std::complex<double>, 1> &c,
                   std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
    omatadd_batch_precondition(selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a,
                               beta, b, ldb, stride_b, c, ldc, stride_c, batch_size);
    oneapi::mkl::blas::mklgpu::MAJOR::omatadd_batch(selector.get_queue(), transa, transb, m, n,
                                                    alpha, a, lda, stride_a, beta, b, ldb, stride_b,
                                                    c, ldc, stride_c, batch_size);
    omatadd_batch_postcondition(selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a,
                                beta, b, ldb, stride_b, c, ldc, stride_c, batch_size);
}

// USM APIs

sycl::event syr2(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     float alpha, const float *x, std::int64_t incx, const float *y,
                     std::int64_t incy, float *a, std::int64_t lda,
                     const std::vector<sycl::event> &dependencies) {
    syr2_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a, lda,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::syr2(selector.get_queue(), upper_lower, n, alpha,
                                                       x, incx, y, incy, a, lda, dependencies);
    syr2_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a, lda,
                       dependencies);
    return done;
}

sycl::event syr2(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     double alpha, const double *x, std::int64_t incx, const double *y,
                     std::int64_t incy, double *a, std::int64_t lda,
                     const std::vector<sycl::event> &dependencies) {
    syr2_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a, lda,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::syr2(selector.get_queue(), upper_lower, n, alpha,
                                                       x, incx, y, incy, a, lda, dependencies);
    syr2_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a, lda,
                       dependencies);
    return done;
}

sycl::event scal(backend_selector<backend::mklgpu> selector, std::int64_t n, float alpha,
                     float *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    scal_precondition(selector.get_queue(), n, alpha, x, incx, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::scal(selector.get_queue(), n, alpha, x, incx,
                                                       dependencies);
    scal_postcondition(selector.get_queue(), n, alpha, x, incx, dependencies);
    return done;
}

sycl::event scal(backend_selector<backend::mklgpu> selector, std::int64_t n, double alpha,
                     double *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    scal_precondition(selector.get_queue(), n, alpha, x, incx, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::scal(selector.get_queue(), n, alpha, x, incx,
                                                       dependencies);
    scal_postcondition(selector.get_queue(), n, alpha, x, incx, dependencies);
    return done;
}

sycl::event scal(backend_selector<backend::mklgpu> selector, std::int64_t n,
                     std::complex<float> alpha, std::complex<float> *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    scal_precondition(selector.get_queue(), n, alpha, x, incx, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::scal(selector.get_queue(), n, alpha, x, incx,
                                                       dependencies);
    scal_postcondition(selector.get_queue(), n, alpha, x, incx, dependencies);
    return done;
}

sycl::event scal(backend_selector<backend::mklgpu> selector, std::int64_t n,
                     std::complex<double> alpha, std::complex<double> *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    scal_precondition(selector.get_queue(), n, alpha, x, incx, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::scal(selector.get_queue(), n, alpha, x, incx,
                                                       dependencies);
    scal_postcondition(selector.get_queue(), n, alpha, x, incx, dependencies);
    return done;
}

sycl::event scal(backend_selector<backend::mklgpu> selector, std::int64_t n, float alpha,
                     std::complex<float> *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    scal_precondition(selector.get_queue(), n, alpha, x, incx, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::scal(selector.get_queue(), n, alpha, x, incx,
                                                       dependencies);
    scal_postcondition(selector.get_queue(), n, alpha, x, incx, dependencies);
    return done;
}

sycl::event scal(backend_selector<backend::mklgpu> selector, std::int64_t n, double alpha,
                     std::complex<double> *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    scal_precondition(selector.get_queue(), n, alpha, x, incx, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::scal(selector.get_queue(), n, alpha, x, incx,
                                                       dependencies);
    scal_postcondition(selector.get_queue(), n, alpha, x, incx, dependencies);
    return done;
}

sycl::event trmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, const float *a, std::int64_t lda, float *x,
                     std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    trmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trmv(selector.get_queue(), upper_lower, trans,
                                                       unit_diag, n, a, lda, x, incx, dependencies);
    trmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx,
                       dependencies);
    return done;
}

sycl::event trmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, const double *a, std::int64_t lda, double *x,
                     std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    trmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trmv(selector.get_queue(), upper_lower, trans,
                                                       unit_diag, n, a, lda, x, incx, dependencies);
    trmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx,
                       dependencies);
    return done;
}

sycl::event trmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    trmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trmv(selector.get_queue(), upper_lower, trans,
                                                       unit_diag, n, a, lda, x, incx, dependencies);
    trmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx,
                       dependencies);
    return done;
}

sycl::event trmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    trmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trmv(selector.get_queue(), upper_lower, trans,
                                                       unit_diag, n, a, lda, x, incx, dependencies);
    trmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx,
                       dependencies);
    return done;
}

sycl::event tpmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, const float *a, float *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    tpmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::tpmv(selector.get_queue(), upper_lower, trans,
                                                       unit_diag, n, a, x, incx, dependencies);
    tpmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx,
                       dependencies);
    return done;
}

sycl::event tpmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, const double *a, double *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    tpmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::tpmv(selector.get_queue(), upper_lower, trans,
                                                       unit_diag, n, a, x, incx, dependencies);
    tpmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx,
                       dependencies);
    return done;
}

sycl::event tpmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, const std::complex<float> *a,
                     std::complex<float> *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    tpmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::tpmv(selector.get_queue(), upper_lower, trans,
                                                       unit_diag, n, a, x, incx, dependencies);
    tpmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx,
                       dependencies);
    return done;
}

sycl::event tpmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, const std::complex<double> *a,
                     std::complex<double> *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    tpmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::tpmv(selector.get_queue(), upper_lower, trans,
                                                       unit_diag, n, a, x, incx, dependencies);
    tpmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx,
                       dependencies);
    return done;
}

sycl::event spr(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                    float alpha, const float *x, std::int64_t incx, float *a,
                    const std::vector<sycl::event> &dependencies) {
    spr_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::spr(selector.get_queue(), upper_lower, n, alpha,
                                                      x, incx, a, dependencies);
    spr_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, dependencies);
    return done;
}

sycl::event spr(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                    double alpha, const double *x, std::int64_t incx, double *a,
                    const std::vector<sycl::event> &dependencies) {
    spr_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::spr(selector.get_queue(), upper_lower, n, alpha,
                                                      x, incx, a, dependencies);
    spr_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, dependencies);
    return done;
}

sycl::event hpmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    hpmv_precondition(selector.get_queue(), upper_lower, n, alpha, a, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::hpmv(selector.get_queue(), upper_lower, n, alpha,
                                                       a, x, incx, beta, y, incy, dependencies);
    hpmv_postcondition(selector.get_queue(), upper_lower, n, alpha, a, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

sycl::event hpmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    hpmv_precondition(selector.get_queue(), upper_lower, n, alpha, a, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::hpmv(selector.get_queue(), upper_lower, n, alpha,
                                                       a, x, incx, beta, y, incy, dependencies);
    hpmv_postcondition(selector.get_queue(), upper_lower, n, alpha, a, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

sycl::event syrk(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
                     float beta, float *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    syrk_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::syrk(
        selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    syrk_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc,
                       dependencies);
    return done;
}

sycl::event syrk(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     std::int64_t n, std::int64_t k, double alpha, const double *a,
                     std::int64_t lda, double beta, double *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    syrk_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::syrk(
        selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    syrk_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc,
                       dependencies);
    return done;
}

sycl::event syrk(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     std::int64_t n, std::int64_t k, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, std::complex<float> beta,
                     std::complex<float> *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    syrk_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::syrk(
        selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    syrk_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc,
                       dependencies);
    return done;
}

sycl::event syrk(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     std::int64_t n, std::int64_t k, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, std::complex<double> beta,
                     std::complex<double> *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    syrk_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::syrk(
        selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    syrk_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc,
                       dependencies);
    return done;
}

sycl::event syrk_batch(backend_selector<backend::mklgpu> selector, uplo *upper_lower,
                           transpose *trans, std::int64_t *n, std::int64_t *k, float *alpha,
                           const float **a, std::int64_t *lda, float *beta, float **c,
                           std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    syrk_batch_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c,
                            ldc, group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::syrk_batch(
        selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, group_count,
        group_size, dependencies);
    syrk_batch_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c,
                             ldc, group_count, group_size, dependencies);
    return done;
}

sycl::event syrk_batch(backend_selector<backend::mklgpu> selector, uplo *upper_lower,
                           transpose *trans, std::int64_t *n, std::int64_t *k, double *alpha,
                           const double **a, std::int64_t *lda, double *beta, double **c,
                           std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    syrk_batch_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c,
                            ldc, group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::syrk_batch(
        selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, group_count,
        group_size, dependencies);
    syrk_batch_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c,
                             ldc, group_count, group_size, dependencies);
    return done;
}

sycl::event syrk_batch(backend_selector<backend::mklgpu> selector, uplo *upper_lower,
                           transpose *trans, std::int64_t *n, std::int64_t *k,
                           std::complex<float> *alpha, const std::complex<float> **a,
                           std::int64_t *lda, std::complex<float> *beta, std::complex<float> **c,
                           std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    syrk_batch_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c,
                            ldc, group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::syrk_batch(
        selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, group_count,
        group_size, dependencies);
    syrk_batch_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c,
                             ldc, group_count, group_size, dependencies);
    return done;
}

sycl::event syrk_batch(backend_selector<backend::mklgpu> selector, uplo *upper_lower,
                           transpose *trans, std::int64_t *n, std::int64_t *k,
                           std::complex<double> *alpha, const std::complex<double> **a,
                           std::int64_t *lda, std::complex<double> *beta, std::complex<double> **c,
                           std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    syrk_batch_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c,
                            ldc, group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::syrk_batch(
        selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, group_count,
        group_size, dependencies);
    syrk_batch_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c,
                             ldc, group_count, group_size, dependencies);
    return done;
}

sycl::event syrk_batch(backend_selector<backend::mklgpu> selector, uplo upper_lower,
                           transpose trans, std::int64_t n, std::int64_t k, float alpha,
                           const float *a, std::int64_t lda, std::int64_t stride_a, float beta,
                           float *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    syrk_batch_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, stride_a,
                            beta, c, ldc, stride_c, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::syrk_batch(
        selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, stride_a, beta, c, ldc,
        stride_c, batch_size, dependencies);
    syrk_batch_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda,
                             stride_a, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

sycl::event syrk_batch(backend_selector<backend::mklgpu> selector, uplo upper_lower,
                           transpose trans, std::int64_t n, std::int64_t k, double alpha,
                           const double *a, std::int64_t lda, std::int64_t stride_a, double beta,
                           double *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    syrk_batch_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, stride_a,
                            beta, c, ldc, stride_c, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::syrk_batch(
        selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, stride_a, beta, c, ldc,
        stride_c, batch_size, dependencies);
    syrk_batch_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda,
                             stride_a, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

sycl::event syrk_batch(backend_selector<backend::mklgpu> selector, uplo upper_lower,
                           transpose trans, std::int64_t n, std::int64_t k,
                           std::complex<float> alpha, const std::complex<float> *a,
                           std::int64_t lda, std::int64_t stride_a, std::complex<float> beta,
                           std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    syrk_batch_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, stride_a,
                            beta, c, ldc, stride_c, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::syrk_batch(
        selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, stride_a, beta, c, ldc,
        stride_c, batch_size, dependencies);
    syrk_batch_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda,
                             stride_a, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

sycl::event syrk_batch(backend_selector<backend::mklgpu> selector, uplo upper_lower,
                           transpose trans, std::int64_t n, std::int64_t k,
                           std::complex<double> alpha, const std::complex<double> *a,
                           std::int64_t lda, std::int64_t stride_a, std::complex<double> beta,
                           std::complex<double> *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    syrk_batch_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, stride_a,
                            beta, c, ldc, stride_c, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::syrk_batch(
        selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, stride_a, beta, c, ldc,
        stride_c, batch_size, dependencies);
    syrk_batch_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda,
                             stride_a, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

sycl::event her2(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda, const std::vector<sycl::event> &dependencies) {
    her2_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a, lda,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::her2(selector.get_queue(), upper_lower, n, alpha,
                                                       x, incx, y, incy, a, lda, dependencies);
    her2_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a, lda,
                       dependencies);
    return done;
}

sycl::event her2(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda, const std::vector<sycl::event> &dependencies) {
    her2_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a, lda,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::her2(selector.get_queue(), upper_lower, n, alpha,
                                                       x, incx, y, incy, a, lda, dependencies);
    her2_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a, lda,
                       dependencies);
    return done;
}

sycl::event hbmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                     std::int64_t lda, const std::complex<float> *x, std::int64_t incx,
                     std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    hbmv_precondition(selector.get_queue(), upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                      incy, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::hbmv(selector.get_queue(), upper_lower, n, k, alpha, a,
                                               lda, x, incx, beta, y, incy, dependencies);
    hbmv_postcondition(selector.get_queue(), upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                       incy, dependencies);
    return done;
}

sycl::event hbmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                     std::int64_t lda, const std::complex<double> *x, std::int64_t incx,
                     std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    hbmv_precondition(selector.get_queue(), upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                      incy, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::hbmv(selector.get_queue(), upper_lower, n, k, alpha, a,
                                               lda, x, incx, beta, y, incy, dependencies);
    hbmv_postcondition(selector.get_queue(), upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                       incy, dependencies);
    return done;
}

sycl::event rot(backend_selector<backend::mklgpu> selector, std::int64_t n,
                    std::complex<float> *x, std::int64_t incx, std::complex<float> *y,
                    std::int64_t incy, float c, float s,
                    const std::vector<sycl::event> &dependencies) {
    rot_precondition(selector.get_queue(), n, x, incx, y, incy, c, s, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::rot(selector.get_queue(), n, x, incx, y, incy, c,
                                                      s, dependencies);
    rot_postcondition(selector.get_queue(), n, x, incx, y, incy, c, s, dependencies);
    return done;
}

sycl::event rot(backend_selector<backend::mklgpu> selector, std::int64_t n,
                    std::complex<double> *x, std::int64_t incx, std::complex<double> *y,
                    std::int64_t incy, double c, double s,
                    const std::vector<sycl::event> &dependencies) {
    rot_precondition(selector.get_queue(), n, x, incx, y, incy, c, s, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::rot(selector.get_queue(), n, x, incx, y, incy, c,
                                                      s, dependencies);
    rot_postcondition(selector.get_queue(), n, x, incx, y, incy, c, s, dependencies);
    return done;
}

sycl::event rot(backend_selector<backend::mklgpu> selector, std::int64_t n, float *x,
                    std::int64_t incx, float *y, std::int64_t incy, float c, float s,
                    const std::vector<sycl::event> &dependencies) {
    rot_precondition(selector.get_queue(), n, x, incx, y, incy, c, s, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::rot(selector.get_queue(), n, x, incx, y, incy, c,
                                                      s, dependencies);
    rot_postcondition(selector.get_queue(), n, x, incx, y, incy, c, s, dependencies);
    return done;
}

sycl::event rot(backend_selector<backend::mklgpu> selector, std::int64_t n, double *x,
                    std::int64_t incx, double *y, std::int64_t incy, double c, double s,
                    const std::vector<sycl::event> &dependencies) {
    rot_precondition(selector.get_queue(), n, x, incx, y, incy, c, s, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::rot(selector.get_queue(), n, x, incx, y, incy, c,
                                                      s, dependencies);
    rot_postcondition(selector.get_queue(), n, x, incx, y, incy, c, s, dependencies);
    return done;
}

sycl::event axpy(backend_selector<backend::mklgpu> selector, std::int64_t n, float alpha,
                     const float *x, std::int64_t incx, float *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    axpy_precondition(selector.get_queue(), n, alpha, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::axpy(selector.get_queue(), n, alpha, x, incx, y,
                                                       incy, dependencies);
    axpy_postcondition(selector.get_queue(), n, alpha, x, incx, y, incy, dependencies);
    return done;
}

sycl::event axpy(backend_selector<backend::mklgpu> selector, std::int64_t n, double alpha,
                     const double *x, std::int64_t incx, double *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    axpy_precondition(selector.get_queue(), n, alpha, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::axpy(selector.get_queue(), n, alpha, x, incx, y,
                                                       incy, dependencies);
    axpy_postcondition(selector.get_queue(), n, alpha, x, incx, y, incy, dependencies);
    return done;
}

sycl::event axpy(backend_selector<backend::mklgpu> selector, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     std::complex<float> *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    axpy_precondition(selector.get_queue(), n, alpha, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::axpy(selector.get_queue(), n, alpha, x, incx, y,
                                                       incy, dependencies);
    axpy_postcondition(selector.get_queue(), n, alpha, x, incx, y, incy, dependencies);
    return done;
}

sycl::event axpy(backend_selector<backend::mklgpu> selector, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     std::complex<double> *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    axpy_precondition(selector.get_queue(), n, alpha, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::axpy(selector.get_queue(), n, alpha, x, incx, y,
                                                       incy, dependencies);
    axpy_postcondition(selector.get_queue(), n, alpha, x, incx, y, incy, dependencies);
    return done;
}

sycl::event axpy_batch(backend_selector<backend::mklgpu> selector, std::int64_t *n,
                           float *alpha, const float **x, std::int64_t *incx, float **y,
                           std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    axpy_batch_precondition(selector.get_queue(), n, alpha, x, incx, y, incy, group_count,
                            group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::axpy_batch(
        selector.get_queue(), n, alpha, x, incx, y, incy, group_count, group_size, dependencies);
    axpy_batch_postcondition(selector.get_queue(), n, alpha, x, incx, y, incy, group_count,
                             group_size, dependencies);
    return done;
}

sycl::event axpy_batch(backend_selector<backend::mklgpu> selector, std::int64_t *n,
                           double *alpha, const double **x, std::int64_t *incx, double **y,
                           std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    axpy_batch_precondition(selector.get_queue(), n, alpha, x, incx, y, incy, group_count,
                            group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::axpy_batch(
        selector.get_queue(), n, alpha, x, incx, y, incy, group_count, group_size, dependencies);
    axpy_batch_postcondition(selector.get_queue(), n, alpha, x, incx, y, incy, group_count,
                             group_size, dependencies);
    return done;
}

sycl::event axpy_batch(backend_selector<backend::mklgpu> selector, std::int64_t *n,
                           std::complex<float> *alpha, const std::complex<float> **x,
                           std::int64_t *incx, std::complex<float> **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    axpy_batch_precondition(selector.get_queue(), n, alpha, x, incx, y, incy, group_count,
                            group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::axpy_batch(
        selector.get_queue(), n, alpha, x, incx, y, incy, group_count, group_size, dependencies);
    axpy_batch_postcondition(selector.get_queue(), n, alpha, x, incx, y, incy, group_count,
                             group_size, dependencies);
    return done;
}

sycl::event axpy_batch(backend_selector<backend::mklgpu> selector, std::int64_t *n,
                           std::complex<double> *alpha, const std::complex<double> **x,
                           std::int64_t *incx, std::complex<double> **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    axpy_batch_precondition(selector.get_queue(), n, alpha, x, incx, y, incy, group_count,
                            group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::axpy_batch(
        selector.get_queue(), n, alpha, x, incx, y, incy, group_count, group_size, dependencies);
    axpy_batch_postcondition(selector.get_queue(), n, alpha, x, incx, y, incy, group_count,
                             group_size, dependencies);
    return done;
}

sycl::event axpy_batch(backend_selector<backend::mklgpu> selector, std::int64_t n, float alpha,
                           const float *x, std::int64_t incx, std::int64_t stridex, float *y,
                           std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    axpy_batch_precondition(selector.get_queue(), n, alpha, x, incx, stridex, y, incy, stridey,
                            batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::axpy_batch(selector.get_queue(), n, alpha, x,
                                                             incx, stridex, y, incy, stridey,
                                                             batch_size, dependencies);
    axpy_batch_postcondition(selector.get_queue(), n, alpha, x, incx, stridex, y, incy, stridey,
                             batch_size, dependencies);
    return done;
}

sycl::event axpy_batch(backend_selector<backend::mklgpu> selector, std::int64_t n, double alpha,
                           const double *x, std::int64_t incx, std::int64_t stridex, double *y,
                           std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    axpy_batch_precondition(selector.get_queue(), n, alpha, x, incx, stridex, y, incy, stridey,
                            batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::axpy_batch(selector.get_queue(), n, alpha, x,
                                                             incx, stridex, y, incy, stridey,
                                                             batch_size, dependencies);
    axpy_batch_postcondition(selector.get_queue(), n, alpha, x, incx, stridex, y, incy, stridey,
                             batch_size, dependencies);
    return done;
}

sycl::event axpy_batch(backend_selector<backend::mklgpu> selector, std::int64_t n,
                           std::complex<float> alpha, const std::complex<float> *x,
                           std::int64_t incx, std::int64_t stridex, std::complex<float> *y,
                           std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    axpy_batch_precondition(selector.get_queue(), n, alpha, x, incx, stridex, y, incy, stridey,
                            batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::axpy_batch(selector.get_queue(), n, alpha, x,
                                                             incx, stridex, y, incy, stridey,
                                                             batch_size, dependencies);
    axpy_batch_postcondition(selector.get_queue(), n, alpha, x, incx, stridex, y, incy, stridey,
                             batch_size, dependencies);
    return done;
}

sycl::event axpy_batch(backend_selector<backend::mklgpu> selector, std::int64_t n,
                           std::complex<double> alpha, const std::complex<double> *x,
                           std::int64_t incx, std::int64_t stridex, std::complex<double> *y,
                           std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    axpy_batch_precondition(selector.get_queue(), n, alpha, x, incx, stridex, y, incy, stridey,
                            batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::axpy_batch(selector.get_queue(), n, alpha, x,
                                                             incx, stridex, y, incy, stridey,
                                                             batch_size, dependencies);
    axpy_batch_postcondition(selector.get_queue(), n, alpha, x, incx, stridex, y, incy, stridey,
                             batch_size, dependencies);
    return done;
}

sycl::event axpby(backend_selector<backend::mklgpu> selector, std::int64_t n, float alpha,
                      const float *x, std::int64_t incx, const float beta, float *y,
                      std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    axpby_precondition(selector.get_queue(), n, alpha, x, incx, beta, y, incy, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::axpby(selector.get_queue(), n, alpha, x, incx,
                                                        beta, y, incy, dependencies);
    axpby_postcondition(selector.get_queue(), n, alpha, x, incx, beta, y, incy, dependencies);
    return done;
}

sycl::event axpby(backend_selector<backend::mklgpu> selector, std::int64_t n, double alpha,
                      const double *x, std::int64_t incx, const double beta, double *y,
                      std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    axpby_precondition(selector.get_queue(), n, alpha, x, incx, beta, y, incy, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::axpby(selector.get_queue(), n, alpha, x, incx,
                                                        beta, y, incy, dependencies);
    axpby_postcondition(selector.get_queue(), n, alpha, x, incx, beta, y, incy, dependencies);
    return done;
}

sycl::event axpby(backend_selector<backend::mklgpu> selector, std::int64_t n,
                      std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                      const std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                      const std::vector<sycl::event> &dependencies) {
    axpby_precondition(selector.get_queue(), n, alpha, x, incx, beta, y, incy, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::axpby(selector.get_queue(), n, alpha, x, incx,
                                                        beta, y, incy, dependencies);
    axpby_postcondition(selector.get_queue(), n, alpha, x, incx, beta, y, incy, dependencies);
    return done;
}

sycl::event axpby(backend_selector<backend::mklgpu> selector, std::int64_t n,
                      std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                      const std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
                      const std::vector<sycl::event> &dependencies) {
    axpby_precondition(selector.get_queue(), n, alpha, x, incx, beta, y, incy, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::axpby(selector.get_queue(), n, alpha, x, incx,
                                                        beta, y, incy, dependencies);
    axpby_postcondition(selector.get_queue(), n, alpha, x, incx, beta, y, incy, dependencies);
    return done;
}

sycl::event gerc(backend_selector<backend::mklgpu> selector, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda, const std::vector<sycl::event> &dependencies) {
    gerc_precondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gerc(selector.get_queue(), m, n, alpha, x, incx,
                                                       y, incy, a, lda, dependencies);
    gerc_postcondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

sycl::event gerc(backend_selector<backend::mklgpu> selector, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda, const std::vector<sycl::event> &dependencies) {
    gerc_precondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gerc(selector.get_queue(), m, n, alpha, x, incx,
                                                       y, incy, a, lda, dependencies);
    gerc_postcondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

sycl::event syr2k(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                      std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
                      const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                      const std::vector<sycl::event> &dependencies) {
    syr2k_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::syr2k(selector.get_queue(), upper_lower, trans, n, k,
                                                alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
    syr2k_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                        c, ldc, dependencies);
    return done;
}

sycl::event syr2k(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                      std::int64_t n, std::int64_t k, double alpha, const double *a,
                      std::int64_t lda, const double *b, std::int64_t ldb, double beta, double *c,
                      std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    syr2k_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::syr2k(selector.get_queue(), upper_lower, trans, n, k,
                                                alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
    syr2k_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                        c, ldc, dependencies);
    return done;
}

sycl::event syr2k(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                      std::int64_t n, std::int64_t k, std::complex<float> alpha,
                      const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                      std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                      std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    syr2k_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::syr2k(selector.get_queue(), upper_lower, trans, n, k,
                                                alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
    syr2k_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                        c, ldc, dependencies);
    return done;
}

sycl::event syr2k(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                      std::int64_t n, std::int64_t k, std::complex<double> alpha,
                      const std::complex<double> *a, std::int64_t lda,
                      const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                      std::complex<double> *c, std::int64_t ldc,
                      const std::vector<sycl::event> &dependencies) {
    syr2k_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::syr2k(selector.get_queue(), upper_lower, trans, n, k,
                                                alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
    syr2k_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                        c, ldc, dependencies);
    return done;
}

sycl::event gemv(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                     std::int64_t n, float alpha, const float *a, std::int64_t lda, const float *x,
                     std::int64_t incx, float beta, float *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    gemv_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemv(selector.get_queue(), trans, m, n, alpha, a,
                                                       lda, x, incx, beta, y, incy, dependencies);
    gemv_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

sycl::event gemv(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                     std::int64_t n, double alpha, const double *a, std::int64_t lda,
                     const double *x, std::int64_t incx, double beta, double *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    gemv_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemv(selector.get_queue(), trans, m, n, alpha, a,
                                                       lda, x, incx, beta, y, incy, dependencies);
    gemv_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

sycl::event gemv(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                     std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                     std::int64_t lda, const std::complex<float> *x, std::int64_t incx,
                     std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    gemv_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemv(selector.get_queue(), trans, m, n, alpha, a,
                                                       lda, x, incx, beta, y, incy, dependencies);
    gemv_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

sycl::event gemv(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                     std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                     std::int64_t lda, const std::complex<double> *x, std::int64_t incx,
                     std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    gemv_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemv(selector.get_queue(), trans, m, n, alpha, a,
                                                       lda, x, incx, beta, y, incy, dependencies);
    gemv_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

sycl::event gemv_batch(backend_selector<backend::mklgpu> selector, transpose trans,
                           std::int64_t m, std::int64_t n, float alpha, const float *a,
                           std::int64_t lda, std::int64_t stridea, const float *x,
                           std::int64_t incx, std::int64_t stridex, float beta, float *y,
                           std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    gemv_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx,
                            stridex, beta, y, incy, stridey, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemv_batch(
        selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx, stridex, beta, y, incy,
        stridey, batch_size, dependencies);
    gemv_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx,
                             stridex, beta, y, incy, stridey, batch_size, dependencies);
    return done;
}

sycl::event gemv_batch(backend_selector<backend::mklgpu> selector, transpose trans,
                           std::int64_t m, std::int64_t n, double alpha, const double *a,
                           std::int64_t lda, std::int64_t stridea, const double *x,
                           std::int64_t incx, std::int64_t stridex, double beta, double *y,
                           std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    gemv_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx,
                            stridex, beta, y, incy, stridey, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemv_batch(
        selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx, stridex, beta, y, incy,
        stridey, batch_size, dependencies);
    gemv_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx,
                             stridex, beta, y, incy, stridey, batch_size, dependencies);
    return done;
}

sycl::event gemv_batch(backend_selector<backend::mklgpu> selector, transpose trans,
                           std::int64_t m, std::int64_t n, std::complex<float> alpha,
                           const std::complex<float> *a, std::int64_t lda, std::int64_t stridea,
                           const std::complex<float> *x, std::int64_t incx, std::int64_t stridex,
                           std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                           std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    gemv_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx,
                            stridex, beta, y, incy, stridey, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemv_batch(
        selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx, stridex, beta, y, incy,
        stridey, batch_size, dependencies);
    gemv_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx,
                             stridex, beta, y, incy, stridey, batch_size, dependencies);
    return done;
}

sycl::event gemv_batch(backend_selector<backend::mklgpu> selector, transpose trans,
                           std::int64_t m, std::int64_t n, std::complex<double> alpha,
                           const std::complex<double> *a, std::int64_t lda, std::int64_t stridea,
                           const std::complex<double> *x, std::int64_t incx, std::int64_t stridex,
                           std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
                           std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    gemv_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx,
                            stridex, beta, y, incy, stridey, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemv_batch(
        selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx, stridex, beta, y, incy,
        stridey, batch_size, dependencies);
    gemv_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, stridea, x, incx,
                             stridex, beta, y, incy, stridey, batch_size, dependencies);
    return done;
}

sycl::event gemv_batch(backend_selector<backend::mklgpu> selector, transpose *trans,
                           std::int64_t *m, std::int64_t *n, float *alpha, const float **a,
                           std::int64_t *lda, const float **x, std::int64_t *incx, float *beta,
                           float **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    gemv_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y,
                            incy, group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemv_batch(selector.get_queue(), trans, m, n,
                                                             alpha, a, lda, x, incx, beta, y, incy,
                                                             group_count, group_size, dependencies);
    gemv_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y,
                             incy, group_count, group_size, dependencies);
    return done;
}

sycl::event gemv_batch(backend_selector<backend::mklgpu> selector, transpose *trans,
                           std::int64_t *m, std::int64_t *n, double *alpha, const double **a,
                           std::int64_t *lda, const double **x, std::int64_t *incx, double *beta,
                           double **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    gemv_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y,
                            incy, group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemv_batch(selector.get_queue(), trans, m, n,
                                                             alpha, a, lda, x, incx, beta, y, incy,
                                                             group_count, group_size, dependencies);
    gemv_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y,
                             incy, group_count, group_size, dependencies);
    return done;
}

sycl::event gemv_batch(backend_selector<backend::mklgpu> selector, transpose *trans,
                           std::int64_t *m, std::int64_t *n, std::complex<float> *alpha,
                           const std::complex<float> **a, std::int64_t *lda,
                           const std::complex<float> **x, std::int64_t *incx,
                           std::complex<float> *beta, std::complex<float> **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    gemv_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y,
                            incy, group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemv_batch(selector.get_queue(), trans, m, n,
                                                             alpha, a, lda, x, incx, beta, y, incy,
                                                             group_count, group_size, dependencies);
    gemv_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y,
                             incy, group_count, group_size, dependencies);
    return done;
}

sycl::event gemv_batch(backend_selector<backend::mklgpu> selector, transpose *trans,
                           std::int64_t *m, std::int64_t *n, std::complex<double> *alpha,
                           const std::complex<double> **a, std::int64_t *lda,
                           const std::complex<double> **x, std::int64_t *incx,
                           std::complex<double> *beta, std::complex<double> **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    gemv_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y,
                            incy, group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemv_batch(selector.get_queue(), trans, m, n,
                                                             alpha, a, lda, x, incx, beta, y, incy,
                                                             group_count, group_size, dependencies);
    gemv_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, x, incx, beta, y,
                             incy, group_count, group_size, dependencies);
    return done;
}

sycl::event dgmm_batch(backend_selector<backend::mklgpu> selector, side left_right,
                           std::int64_t m, std::int64_t n, const float *a, std::int64_t lda,
                           std::int64_t stridea, const float *x, std::int64_t incx,
                           std::int64_t stridex, float *c, std::int64_t ldc, std::int64_t stridec,
                           std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    dgmm_batch_precondition(selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx,
                            stridex, c, ldc, stridec, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::dgmm_batch(
        selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx, stridex, c, ldc, stridec,
        batch_size, dependencies);
    dgmm_batch_postcondition(selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx,
                             stridex, c, ldc, stridec, batch_size, dependencies);
    return done;
}

sycl::event dgmm_batch(backend_selector<backend::mklgpu> selector, side left_right,
                           std::int64_t m, std::int64_t n, const double *a, std::int64_t lda,
                           std::int64_t stridea, const double *x, std::int64_t incx,
                           std::int64_t stridex, double *c, std::int64_t ldc, std::int64_t stridec,
                           std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    dgmm_batch_precondition(selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx,
                            stridex, c, ldc, stridec, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::dgmm_batch(
        selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx, stridex, c, ldc, stridec,
        batch_size, dependencies);
    dgmm_batch_postcondition(selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx,
                             stridex, c, ldc, stridec, batch_size, dependencies);
    return done;
}

sycl::event dgmm_batch(backend_selector<backend::mklgpu> selector, side left_right,
                           std::int64_t m, std::int64_t n, const std::complex<float> *a,
                           std::int64_t lda, std::int64_t stridea, const std::complex<float> *x,
                           std::int64_t incx, std::int64_t stridex, std::complex<float> *c,
                           std::int64_t ldc, std::int64_t stridec, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    dgmm_batch_precondition(selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx,
                            stridex, c, ldc, stridec, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::dgmm_batch(
        selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx, stridex, c, ldc, stridec,
        batch_size, dependencies);
    dgmm_batch_postcondition(selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx,
                             stridex, c, ldc, stridec, batch_size, dependencies);
    return done;
}

sycl::event dgmm_batch(backend_selector<backend::mklgpu> selector, side left_right,
                           std::int64_t m, std::int64_t n, const std::complex<double> *a,
                           std::int64_t lda, std::int64_t stridea, const std::complex<double> *x,
                           std::int64_t incx, std::int64_t stridex, std::complex<double> *c,
                           std::int64_t ldc, std::int64_t stridec, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    dgmm_batch_precondition(selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx,
                            stridex, c, ldc, stridec, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::dgmm_batch(
        selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx, stridex, c, ldc, stridec,
        batch_size, dependencies);
    dgmm_batch_postcondition(selector.get_queue(), left_right, m, n, a, lda, stridea, x, incx,
                             stridex, c, ldc, stridec, batch_size, dependencies);
    return done;
}

sycl::event dgmm_batch(backend_selector<backend::mklgpu> selector, side *left_right,
                           std::int64_t *m, std::int64_t *n, const float **a, std::int64_t *lda,
                           const float **x, std::int64_t *incx, float **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    dgmm_batch_precondition(selector.get_queue(), left_right, m, n, a, lda, x, incx, c, ldc,
                            group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::dgmm_batch(selector.get_queue(), left_right, m, n,
                                                             a, lda, x, incx, c, ldc, group_count,
                                                             group_size, dependencies);
    dgmm_batch_postcondition(selector.get_queue(), left_right, m, n, a, lda, x, incx, c, ldc,
                             group_count, group_size, dependencies);
    return done;
}

sycl::event dgmm_batch(backend_selector<backend::mklgpu> selector, side *left_right,
                           std::int64_t *m, std::int64_t *n, const double **a, std::int64_t *lda,
                           const double **x, std::int64_t *incx, double **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    dgmm_batch_precondition(selector.get_queue(), left_right, m, n, a, lda, x, incx, c, ldc,
                            group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::dgmm_batch(selector.get_queue(), left_right, m, n,
                                                             a, lda, x, incx, c, ldc, group_count,
                                                             group_size, dependencies);
    dgmm_batch_postcondition(selector.get_queue(), left_right, m, n, a, lda, x, incx, c, ldc,
                             group_count, group_size, dependencies);
    return done;
}

sycl::event dgmm_batch(backend_selector<backend::mklgpu> selector, side *left_right,
                           std::int64_t *m, std::int64_t *n, const std::complex<float> **a,
                           std::int64_t *lda, const std::complex<float> **x, std::int64_t *incx,
                           std::complex<float> **c, std::int64_t *ldc, std::int64_t group_count,
                           std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    dgmm_batch_precondition(selector.get_queue(), left_right, m, n, a, lda, x, incx, c, ldc,
                            group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::dgmm_batch(selector.get_queue(), left_right, m, n,
                                                             a, lda, x, incx, c, ldc, group_count,
                                                             group_size, dependencies);
    dgmm_batch_postcondition(selector.get_queue(), left_right, m, n, a, lda, x, incx, c, ldc,
                             group_count, group_size, dependencies);
    return done;
}

sycl::event dgmm_batch(backend_selector<backend::mklgpu> selector, side *left_right,
                           std::int64_t *m, std::int64_t *n, const std::complex<double> **a,
                           std::int64_t *lda, const std::complex<double> **x, std::int64_t *incx,
                           std::complex<double> **c, std::int64_t *ldc, std::int64_t group_count,
                           std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    dgmm_batch_precondition(selector.get_queue(), left_right, m, n, a, lda, x, incx, c, ldc,
                            group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::dgmm_batch(selector.get_queue(), left_right, m, n,
                                                             a, lda, x, incx, c, ldc, group_count,
                                                             group_size, dependencies);
    dgmm_batch_postcondition(selector.get_queue(), left_right, m, n, a, lda, x, incx, c, ldc,
                             group_count, group_size, dependencies);
    return done;
}

sycl::event her(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                    float alpha, const std::complex<float> *x, std::int64_t incx,
                    std::complex<float> *a, std::int64_t lda,
                    const std::vector<sycl::event> &dependencies) {
    her_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, lda, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::her(selector.get_queue(), upper_lower, n, alpha,
                                                      x, incx, a, lda, dependencies);
    her_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, lda, dependencies);
    return done;
}

sycl::event her(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                    double alpha, const std::complex<double> *x, std::int64_t incx,
                    std::complex<double> *a, std::int64_t lda,
                    const std::vector<sycl::event> &dependencies) {
    her_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, lda, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::her(selector.get_queue(), upper_lower, n, alpha,
                                                      x, incx, a, lda, dependencies);
    her_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, lda, dependencies);
    return done;
}

sycl::event hpr(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                    float alpha, const std::complex<float> *x, std::int64_t incx,
                    std::complex<float> *a, const std::vector<sycl::event> &dependencies) {
    hpr_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::hpr(selector.get_queue(), upper_lower, n, alpha,
                                                      x, incx, a, dependencies);
    hpr_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, dependencies);
    return done;
}

sycl::event hpr(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                    double alpha, const std::complex<double> *x, std::int64_t incx,
                    std::complex<double> *a, const std::vector<sycl::event> &dependencies) {
    hpr_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::hpr(selector.get_queue(), upper_lower, n, alpha,
                                                      x, incx, a, dependencies);
    hpr_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, dependencies);
    return done;
}

sycl::event iamin(backend_selector<backend::mklgpu> selector, std::int64_t n, const float *x,
                      std::int64_t incx, std::int64_t *result,
                      const std::vector<sycl::event> &dependencies) {
    iamin_precondition(selector.get_queue(), n, x, incx, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::iamin(selector.get_queue(), n, x, incx, result,
                                                        dependencies);
    iamin_postcondition(selector.get_queue(), n, x, incx, result, dependencies);
    return done;
}

sycl::event iamin(backend_selector<backend::mklgpu> selector, std::int64_t n, const double *x,
                      std::int64_t incx, std::int64_t *result,
                      const std::vector<sycl::event> &dependencies) {
    iamin_precondition(selector.get_queue(), n, x, incx, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::iamin(selector.get_queue(), n, x, incx, result,
                                                        dependencies);
    iamin_postcondition(selector.get_queue(), n, x, incx, result, dependencies);
    return done;
}

sycl::event iamin(backend_selector<backend::mklgpu> selector, std::int64_t n,
                      const std::complex<float> *x, std::int64_t incx, std::int64_t *result,
                      const std::vector<sycl::event> &dependencies) {
    iamin_precondition(selector.get_queue(), n, x, incx, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::iamin(selector.get_queue(), n, x, incx, result,
                                                        dependencies);
    iamin_postcondition(selector.get_queue(), n, x, incx, result, dependencies);
    return done;
}

sycl::event iamin(backend_selector<backend::mklgpu> selector, std::int64_t n,
                      const std::complex<double> *x, std::int64_t incx, std::int64_t *result,
                      const std::vector<sycl::event> &dependencies) {
    iamin_precondition(selector.get_queue(), n, x, incx, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::iamin(selector.get_queue(), n, x, incx, result,
                                                        dependencies);
    iamin_postcondition(selector.get_queue(), n, x, incx, result, dependencies);
    return done;
}

sycl::event gemm_batch(backend_selector<backend::mklgpu> selector, transpose *transa,
                           transpose *transb, std::int64_t *m, std::int64_t *n, std::int64_t *k,
                           sycl::half *alpha, const sycl::half **a, std::int64_t *lda,
                           const sycl::half **b, std::int64_t *ldb, sycl::half *beta,
                           sycl::half **c, std::int64_t *ldc, std::int64_t group_count,
                           std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    gemm_batch_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb,
                            beta, c, ldc, group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemm_batch(
        selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        group_count, group_size, dependencies);
    gemm_batch_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb,
                             beta, c, ldc, group_count, group_size, dependencies);
    return done;
}

sycl::event gemm_batch(backend_selector<backend::mklgpu> selector, transpose *transa,
                           transpose *transb, std::int64_t *m, std::int64_t *n, std::int64_t *k,
                           float *alpha, const float **a, std::int64_t *lda, const float **b,
                           std::int64_t *ldb, float *beta, float **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    gemm_batch_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb,
                            beta, c, ldc, group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemm_batch(
        selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        group_count, group_size, dependencies);
    gemm_batch_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb,
                             beta, c, ldc, group_count, group_size, dependencies);
    return done;
}

sycl::event gemm_batch(backend_selector<backend::mklgpu> selector, transpose *transa,
                           transpose *transb, std::int64_t *m, std::int64_t *n, std::int64_t *k,
                           double *alpha, const double **a, std::int64_t *lda, const double **b,
                           std::int64_t *ldb, double *beta, double **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    gemm_batch_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb,
                            beta, c, ldc, group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemm_batch(
        selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        group_count, group_size, dependencies);
    gemm_batch_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb,
                             beta, c, ldc, group_count, group_size, dependencies);
    return done;
}

sycl::event gemm_batch(backend_selector<backend::mklgpu> selector, transpose *transa,
                           transpose *transb, std::int64_t *m, std::int64_t *n, std::int64_t *k,
                           std::complex<float> *alpha, const std::complex<float> **a,
                           std::int64_t *lda, const std::complex<float> **b, std::int64_t *ldb,
                           std::complex<float> *beta, std::complex<float> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    gemm_batch_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb,
                            beta, c, ldc, group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemm_batch(
        selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        group_count, group_size, dependencies);
    gemm_batch_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb,
                             beta, c, ldc, group_count, group_size, dependencies);
    return done;
}

sycl::event gemm_batch(backend_selector<backend::mklgpu> selector, transpose *transa,
                           transpose *transb, std::int64_t *m, std::int64_t *n, std::int64_t *k,
                           std::complex<double> *alpha, const std::complex<double> **a,
                           std::int64_t *lda, const std::complex<double> **b, std::int64_t *ldb,
                           std::complex<double> *beta, std::complex<double> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    gemm_batch_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb,
                            beta, c, ldc, group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemm_batch(
        selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        group_count, group_size, dependencies);
    gemm_batch_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb,
                             beta, c, ldc, group_count, group_size, dependencies);
    return done;
}

sycl::event gemm_batch(backend_selector<backend::mklgpu> selector, transpose transa,
                           transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                           sycl::half alpha, const sycl::half *a, std::int64_t lda,
                           std::int64_t stride_a, const sycl::half *b, std::int64_t ldb,
                           std::int64_t stride_b, sycl::half beta, sycl::half *c, std::int64_t ldc,
                           std::int64_t stride_c, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    gemm_batch_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                            b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemm_batch(
        selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size, dependencies);
    gemm_batch_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                             b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

sycl::event gemm_batch(backend_selector<backend::mklgpu> selector, transpose transa,
                           transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                           float alpha, const float *a, std::int64_t lda, std::int64_t stride_a,
                           const float *b, std::int64_t ldb, std::int64_t stride_b, float beta,
                           float *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    gemm_batch_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                            b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemm_batch(
        selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size, dependencies);
    gemm_batch_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                             b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

sycl::event gemm_batch(backend_selector<backend::mklgpu> selector, transpose transa,
                           transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                           double alpha, const double *a, std::int64_t lda, std::int64_t stride_a,
                           const double *b, std::int64_t ldb, std::int64_t stride_b, double beta,
                           double *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    gemm_batch_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                            b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemm_batch(
        selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size, dependencies);
    gemm_batch_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                             b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

sycl::event gemm_batch(backend_selector<backend::mklgpu> selector, transpose transa,
                           transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<float> alpha, const std::complex<float> *a,
                           std::int64_t lda, std::int64_t stride_a, const std::complex<float> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                           std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    gemm_batch_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                            b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemm_batch(
        selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size, dependencies);
    gemm_batch_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                             b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

sycl::event gemm_batch(backend_selector<backend::mklgpu> selector, transpose transa,
                           transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<double> alpha, const std::complex<double> *a,
                           std::int64_t lda, std::int64_t stride_a, const std::complex<double> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                           std::complex<double> *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    gemm_batch_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                            b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemm_batch(
        selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size, dependencies);
    gemm_batch_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, stride_a,
                             b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

sycl::event spmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     float alpha, const float *a, const float *x, std::int64_t incx, float beta,
                     float *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    spmv_precondition(selector.get_queue(), upper_lower, n, alpha, a, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::spmv(selector.get_queue(), upper_lower, n, alpha,
                                                       a, x, incx, beta, y, incy, dependencies);
    spmv_postcondition(selector.get_queue(), upper_lower, n, alpha, a, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

sycl::event spmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     double alpha, const double *a, const double *x, std::int64_t incx, double beta,
                     double *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    spmv_precondition(selector.get_queue(), upper_lower, n, alpha, a, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::spmv(selector.get_queue(), upper_lower, n, alpha,
                                                       a, x, incx, beta, y, incy, dependencies);
    spmv_postcondition(selector.get_queue(), upper_lower, n, alpha, a, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

sycl::event swap(backend_selector<backend::mklgpu> selector, std::int64_t n, float *x,
                     std::int64_t incx, float *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    swap_precondition(selector.get_queue(), n, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::swap(selector.get_queue(), n, x, incx, y, incy,
                                                       dependencies);
    swap_postcondition(selector.get_queue(), n, x, incx, y, incy, dependencies);
    return done;
}

sycl::event swap(backend_selector<backend::mklgpu> selector, std::int64_t n, double *x,
                     std::int64_t incx, double *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    swap_precondition(selector.get_queue(), n, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::swap(selector.get_queue(), n, x, incx, y, incy,
                                                       dependencies);
    swap_postcondition(selector.get_queue(), n, x, incx, y, incy, dependencies);
    return done;
}

sycl::event swap(backend_selector<backend::mklgpu> selector, std::int64_t n,
                     std::complex<float> *x, std::int64_t incx, std::complex<float> *y,
                     std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    swap_precondition(selector.get_queue(), n, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::swap(selector.get_queue(), n, x, incx, y, incy,
                                                       dependencies);
    swap_postcondition(selector.get_queue(), n, x, incx, y, incy, dependencies);
    return done;
}

sycl::event swap(backend_selector<backend::mklgpu> selector, std::int64_t n,
                     std::complex<double> *x, std::int64_t incx, std::complex<double> *y,
                     std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    swap_precondition(selector.get_queue(), n, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::swap(selector.get_queue(), n, x, incx, y, incy,
                                                       dependencies);
    swap_postcondition(selector.get_queue(), n, x, incx, y, incy, dependencies);
    return done;
}

sycl::event geru(backend_selector<backend::mklgpu> selector, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda, const std::vector<sycl::event> &dependencies) {
    geru_precondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::geru(selector.get_queue(), m, n, alpha, x, incx,
                                                       y, incy, a, lda, dependencies);
    geru_postcondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

sycl::event geru(backend_selector<backend::mklgpu> selector, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda, const std::vector<sycl::event> &dependencies) {
    geru_precondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::geru(selector.get_queue(), m, n, alpha, x, incx,
                                                       y, incy, a, lda, dependencies);
    geru_postcondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

sycl::event nrm2(backend_selector<backend::mklgpu> selector, std::int64_t n,
                     const std::complex<float> *x, std::int64_t incx, float *result,
                     const std::vector<sycl::event> &dependencies) {
    nrm2_precondition(selector.get_queue(), n, x, incx, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::nrm2(selector.get_queue(), n, x, incx, result,
                                                       dependencies);
    nrm2_postcondition(selector.get_queue(), n, x, incx, result, dependencies);
    return done;
}

sycl::event nrm2(backend_selector<backend::mklgpu> selector, std::int64_t n,
                     const std::complex<double> *x, std::int64_t incx, double *result,
                     const std::vector<sycl::event> &dependencies) {
    nrm2_precondition(selector.get_queue(), n, x, incx, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::nrm2(selector.get_queue(), n, x, incx, result,
                                                       dependencies);
    nrm2_postcondition(selector.get_queue(), n, x, incx, result, dependencies);
    return done;
}

sycl::event nrm2(backend_selector<backend::mklgpu> selector, std::int64_t n, const float *x,
                     std::int64_t incx, float *result,
                     const std::vector<sycl::event> &dependencies) {
    nrm2_precondition(selector.get_queue(), n, x, incx, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::nrm2(selector.get_queue(), n, x, incx, result,
                                                       dependencies);
    nrm2_postcondition(selector.get_queue(), n, x, incx, result, dependencies);
    return done;
}

sycl::event nrm2(backend_selector<backend::mklgpu> selector, std::int64_t n, const double *x,
                     std::int64_t incx, double *result,
                     const std::vector<sycl::event> &dependencies) {
    nrm2_precondition(selector.get_queue(), n, x, incx, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::nrm2(selector.get_queue(), n, x, incx, result,
                                                       dependencies);
    nrm2_postcondition(selector.get_queue(), n, x, incx, result, dependencies);
    return done;
}

sycl::event gemm(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
                     std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float *a,
                     std::int64_t lda, const float *b, std::int64_t ldb, float beta, float *c,
                     std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    gemm_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                      ldc, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::gemm(selector.get_queue(), transa, transb, m, n, k, alpha,
                                               a, lda, b, ldb, beta, c, ldc, dependencies);
    gemm_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc, dependencies);
    return done;
}

sycl::event gemm(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
                     std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double *a,
                     std::int64_t lda, const double *b, std::int64_t ldb, double beta, double *c,
                     std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    gemm_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                      ldc, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::gemm(selector.get_queue(), transa, transb, m, n, k, alpha,
                                               a, lda, b, ldb, beta, c, ldc, dependencies);
    gemm_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc, dependencies);
    return done;
}

sycl::event gemm(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
                     std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                     std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                     std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    gemm_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                      ldc, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::gemm(selector.get_queue(), transa, transb, m, n, k, alpha,
                                               a, lda, b, ldb, beta, c, ldc, dependencies);
    gemm_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc, dependencies);
    return done;
}

sycl::event gemm(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
                     std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
                     std::int64_t ldb, std::complex<double> beta, std::complex<double> *c,
                     std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    gemm_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                      ldc, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::gemm(selector.get_queue(), transa, transb, m, n, k, alpha,
                                               a, lda, b, ldb, beta, c, ldc, dependencies);
    gemm_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc, dependencies);
    return done;
}

sycl::event gemm(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
                     std::int64_t m, std::int64_t n, std::int64_t k, sycl::half alpha,
                     const sycl::half *a, std::int64_t lda, const sycl::half *b, std::int64_t ldb,
                     sycl::half beta, sycl::half *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    gemm_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                      ldc, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::gemm(selector.get_queue(), transa, transb, m, n, k, alpha,
                                               a, lda, b, ldb, beta, c, ldc, dependencies);
    gemm_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc, dependencies);
    return done;
}

sycl::event gemm(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
                     std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                     const sycl::half *a, std::int64_t lda, const sycl::half *b, std::int64_t ldb,
                     float beta, float *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    gemm_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                      ldc, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::gemm(selector.get_queue(), transa, transb, m, n, k, alpha,
                                               a, lda, b, ldb, beta, c, ldc, dependencies);
    gemm_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc, dependencies);
    return done;
}

sycl::event gemm(backend_selector<backend::mklgpu> selector, transpose transa, transpose transb,
                     std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const bfloat16 *a,
                     std::int64_t lda, const bfloat16 *b, std::int64_t ldb, float beta, float *c,
                     std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    gemm_precondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                      ldc, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::gemm(selector.get_queue(), transa, transb, m, n, k, alpha,
                                               a, lda, b, ldb, beta, c, ldc, dependencies);
    gemm_postcondition(selector.get_queue(), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc, dependencies);
    return done;
}

sycl::event gemm_bias(backend_selector<backend::mklgpu> selector, transpose transa,
                          transpose transb, offset offsetc, std::int64_t m, std::int64_t n,
                          std::int64_t k, float alpha, const std::int8_t *a, std::int64_t lda,
                          std::int8_t ao, const std::uint8_t *b, std::int64_t ldb, std::uint8_t bo,
                          float beta, std::int32_t *c, std::int64_t ldc, const std::int32_t *co,
                          const std::vector<sycl::event> &dependencies) {
    gemm_bias_precondition(selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda,
                           ao, b, ldb, bo, beta, c, ldc, co, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemm_bias(
        selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb, bo, beta,
        c, ldc, co, dependencies);
    gemm_bias_postcondition(selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda,
                            ao, b, ldb, bo, beta, c, ldc, co, dependencies);
    return done;
}

sycl::event gemm_bias(backend_selector<backend::mklgpu> selector, transpose transa,
                          transpose transb, offset offsetc, std::int64_t m, std::int64_t n,
                          std::int64_t k, float alpha, const std::int8_t *a, std::int64_t lda,
                          std::int8_t ao, const std::int8_t *b, std::int64_t ldb, std::int8_t bo,
                          float beta, std::int32_t *c, std::int64_t ldc, const std::int32_t *co,
                          const std::vector<sycl::event> &dependencies) {
    gemm_bias_precondition(selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda,
                           ao, b, ldb, bo, beta, c, ldc, co, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemm_bias(
        selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb, bo, beta,
        c, ldc, co, dependencies);
    gemm_bias_postcondition(selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda,
                            ao, b, ldb, bo, beta, c, ldc, co, dependencies);
    return done;
}

sycl::event gemm_bias(backend_selector<backend::mklgpu> selector, transpose transa,
                          transpose transb, offset offsetc, std::int64_t m, std::int64_t n,
                          std::int64_t k, float alpha, const std::uint8_t *a, std::int64_t lda,
                          std::uint8_t ao, const std::int8_t *b, std::int64_t ldb, std::int8_t bo,
                          float beta, std::int32_t *c, std::int64_t ldc, const std::int32_t *co,
                          const std::vector<sycl::event> &dependencies) {
    gemm_bias_precondition(selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda,
                           ao, b, ldb, bo, beta, c, ldc, co, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemm_bias(
        selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb, bo, beta,
        c, ldc, co, dependencies);
    gemm_bias_postcondition(selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda,
                            ao, b, ldb, bo, beta, c, ldc, co, dependencies);
    return done;
}

sycl::event gemm_bias(backend_selector<backend::mklgpu> selector, transpose transa,
                          transpose transb, offset offsetc, std::int64_t m, std::int64_t n,
                          std::int64_t k, float alpha, const std::uint8_t *a, std::int64_t lda,
                          std::uint8_t ao, const std::uint8_t *b, std::int64_t ldb, std::uint8_t bo,
                          float beta, std::int32_t *c, std::int64_t ldc, const std::int32_t *co,
                          const std::vector<sycl::event> &dependencies) {
    gemm_bias_precondition(selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda,
                           ao, b, ldb, bo, beta, c, ldc, co, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemm_bias(
        selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb, bo, beta,
        c, ldc, co, dependencies);
    gemm_bias_postcondition(selector.get_queue(), transa, transb, offsetc, m, n, k, alpha, a, lda,
                            ao, b, ldb, bo, beta, c, ldc, co, dependencies);
    return done;
}

sycl::event herk(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     std::int64_t n, std::int64_t k, float alpha, const std::complex<float> *a,
                     std::int64_t lda, float beta, std::complex<float> *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    herk_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::herk(
        selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    herk_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc,
                       dependencies);
    return done;
}

sycl::event herk(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     std::int64_t n, std::int64_t k, double alpha, const std::complex<double> *a,
                     std::int64_t lda, double beta, std::complex<double> *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    herk_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::herk(
        selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    herk_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc,
                       dependencies);
    return done;
}

sycl::event ger(backend_selector<backend::mklgpu> selector, std::int64_t m, std::int64_t n,
                    float alpha, const float *x, std::int64_t incx, const float *y,
                    std::int64_t incy, float *a, std::int64_t lda,
                    const std::vector<sycl::event> &dependencies) {
    ger_precondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::ger(selector.get_queue(), m, n, alpha, x, incx, y,
                                                      incy, a, lda, dependencies);
    ger_postcondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

sycl::event ger(backend_selector<backend::mklgpu> selector, std::int64_t m, std::int64_t n,
                    double alpha, const double *x, std::int64_t incx, const double *y,
                    std::int64_t incy, double *a, std::int64_t lda,
                    const std::vector<sycl::event> &dependencies) {
    ger_precondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::ger(selector.get_queue(), m, n, alpha, x, incx, y,
                                                      incy, a, lda, dependencies);
    ger_postcondition(selector.get_queue(), m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

sycl::event trsm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                     const float *a, std::int64_t lda, float *b, std::int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    trsm_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                      a, lda, b, ldb, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trsm(selector.get_queue(), left_right,
                                                       upper_lower, trans, unit_diag, m, n, alpha,
                                                       a, lda, b, ldb, dependencies);
    trsm_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                       a, lda, b, ldb, dependencies);
    return done;
}

sycl::event trsm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                     const double *a, std::int64_t lda, double *b, std::int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    trsm_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                      a, lda, b, ldb, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trsm(selector.get_queue(), left_right,
                                                       upper_lower, trans, unit_diag, m, n, alpha,
                                                       a, lda, b, ldb, dependencies);
    trsm_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                       a, lda, b, ldb, dependencies);
    return done;
}

sycl::event trsm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> *b, std::int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    trsm_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                      a, lda, b, ldb, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trsm(selector.get_queue(), left_right,
                                                       upper_lower, trans, unit_diag, m, n, alpha,
                                                       a, lda, b, ldb, dependencies);
    trsm_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                       a, lda, b, ldb, dependencies);
    return done;
}

sycl::event trsm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     std::complex<double> *b, std::int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    trsm_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                      a, lda, b, ldb, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trsm(selector.get_queue(), left_right,
                                                       upper_lower, trans, unit_diag, m, n, alpha,
                                                       a, lda, b, ldb, dependencies);
    trsm_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                       a, lda, b, ldb, dependencies);
    return done;
}

sycl::event trsm_batch(backend_selector<backend::mklgpu> selector, side left_right,
                           uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                           std::int64_t n, float alpha, const float *a, std::int64_t lda,
                           std::int64_t stride_a, float *b, std::int64_t ldb, std::int64_t stride_b,
                           std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    trsm_batch_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                            alpha, a, lda, stride_a, b, ldb, stride_b, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trsm_batch(
        selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
        stride_a, b, ldb, stride_b, batch_size, dependencies);
    trsm_batch_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                             alpha, a, lda, stride_a, b, ldb, stride_b, batch_size, dependencies);
    return done;
}

sycl::event trsm_batch(backend_selector<backend::mklgpu> selector, side left_right,
                           uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                           std::int64_t n, double alpha, const double *a, std::int64_t lda,
                           std::int64_t stride_a, double *b, std::int64_t ldb,
                           std::int64_t stride_b, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    trsm_batch_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                            alpha, a, lda, stride_a, b, ldb, stride_b, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trsm_batch(
        selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
        stride_a, b, ldb, stride_b, batch_size, dependencies);
    trsm_batch_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                             alpha, a, lda, stride_a, b, ldb, stride_b, batch_size, dependencies);
    return done;
}

sycl::event trsm_batch(backend_selector<backend::mklgpu> selector, side left_right,
                           uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                           std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                           std::int64_t lda, std::int64_t stride_a, std::complex<float> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    trsm_batch_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                            alpha, a, lda, stride_a, b, ldb, stride_b, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trsm_batch(
        selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
        stride_a, b, ldb, stride_b, batch_size, dependencies);
    trsm_batch_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                             alpha, a, lda, stride_a, b, ldb, stride_b, batch_size, dependencies);
    return done;
}

sycl::event trsm_batch(backend_selector<backend::mklgpu> selector, side left_right,
                           uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                           std::int64_t n, std::complex<double> alpha,
                           const std::complex<double> *a, std::int64_t lda, std::int64_t stride_a,
                           std::complex<double> *b, std::int64_t ldb, std::int64_t stride_b,
                           std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    trsm_batch_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                            alpha, a, lda, stride_a, b, ldb, stride_b, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trsm_batch(
        selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
        stride_a, b, ldb, stride_b, batch_size, dependencies);
    trsm_batch_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                             alpha, a, lda, stride_a, b, ldb, stride_b, batch_size, dependencies);
    return done;
}

sycl::event trsm_batch(backend_selector<backend::mklgpu> selector, side *left_right,
                           uplo *upper_lower, transpose *trans, diag *unit_diag, std::int64_t *m,
                           std::int64_t *n, float *alpha, const float **a, std::int64_t *lda,
                           float **b, std::int64_t *ldb, std::int64_t group_count,
                           std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    trsm_batch_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                            alpha, a, lda, b, ldb, group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trsm_batch(
        selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
        ldb, group_count, group_size, dependencies);
    trsm_batch_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                             alpha, a, lda, b, ldb, group_count, group_size, dependencies);
    return done;
}

sycl::event trsm_batch(backend_selector<backend::mklgpu> selector, side *left_right,
                           uplo *upper_lower, transpose *trans, diag *unit_diag, std::int64_t *m,
                           std::int64_t *n, double *alpha, const double **a, std::int64_t *lda,
                           double **b, std::int64_t *ldb, std::int64_t group_count,
                           std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    trsm_batch_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                            alpha, a, lda, b, ldb, group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trsm_batch(
        selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
        ldb, group_count, group_size, dependencies);
    trsm_batch_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                             alpha, a, lda, b, ldb, group_count, group_size, dependencies);
    return done;
}

sycl::event trsm_batch(backend_selector<backend::mklgpu> selector, side *left_right,
                           uplo *upper_lower, transpose *trans, diag *unit_diag, std::int64_t *m,
                           std::int64_t *n, std::complex<float> *alpha,
                           const std::complex<float> **a, std::int64_t *lda,
                           std::complex<float> **b, std::int64_t *ldb, std::int64_t group_count,
                           std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    trsm_batch_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                            alpha, a, lda, b, ldb, group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trsm_batch(
        selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
        ldb, group_count, group_size, dependencies);
    trsm_batch_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                             alpha, a, lda, b, ldb, group_count, group_size, dependencies);
    return done;
}

sycl::event trsm_batch(backend_selector<backend::mklgpu> selector, side *left_right,
                           uplo *upper_lower, transpose *trans, diag *unit_diag, std::int64_t *m,
                           std::int64_t *n, std::complex<double> *alpha,
                           const std::complex<double> **a, std::int64_t *lda,
                           std::complex<double> **b, std::int64_t *ldb, std::int64_t group_count,
                           std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    trsm_batch_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                            alpha, a, lda, b, ldb, group_count, group_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trsm_batch(
        selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
        ldb, group_count, group_size, dependencies);
    trsm_batch_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n,
                             alpha, a, lda, b, ldb, group_count, group_size, dependencies);
    return done;
}

sycl::event dotu(backend_selector<backend::mklgpu> selector, std::int64_t n,
                     const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
                     std::int64_t incy, std::complex<float> *result,
                     const std::vector<sycl::event> &dependencies) {
    dotu_precondition(selector.get_queue(), n, x, incx, y, incy, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::dotu(selector.get_queue(), n, x, incx, y, incy,
                                                       result, dependencies);
    dotu_postcondition(selector.get_queue(), n, x, incx, y, incy, result, dependencies);
    return done;
}

sycl::event dotu(backend_selector<backend::mklgpu> selector, std::int64_t n,
                     const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *result,
                     const std::vector<sycl::event> &dependencies) {
    dotu_precondition(selector.get_queue(), n, x, incx, y, incy, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::dotu(selector.get_queue(), n, x, incx, y, incy,
                                                       result, dependencies);
    dotu_postcondition(selector.get_queue(), n, x, incx, y, incy, result, dependencies);
    return done;
}

sycl::event hemm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
                     std::int64_t m, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                     std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                     std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    hemm_precondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                      beta, c, ldc, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::hemm(selector.get_queue(), left_right, upper_lower, m, n,
                                               alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
    hemm_postcondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                       beta, c, ldc, dependencies);
    return done;
}

sycl::event hemm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
                     std::int64_t m, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
                     std::int64_t ldb, std::complex<double> beta, std::complex<double> *c,
                     std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    hemm_precondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                      beta, c, ldc, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::hemm(selector.get_queue(), left_right, upper_lower, m, n,
                                               alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
    hemm_postcondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                       beta, c, ldc, dependencies);
    return done;
}

sycl::event hpr2(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     const std::vector<sycl::event> &dependencies) {
    hpr2_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::hpr2(selector.get_queue(), upper_lower, n, alpha,
                                                       x, incx, y, incy, a, dependencies);
    hpr2_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a,
                       dependencies);
    return done;
}

sycl::event hpr2(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     const std::vector<sycl::event> &dependencies) {
    hpr2_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::hpr2(selector.get_queue(), upper_lower, n, alpha,
                                                       x, incx, y, incy, a, dependencies);
    hpr2_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a,
                       dependencies);
    return done;
}

sycl::event gbmv(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha, const float *a,
                     std::int64_t lda, const float *x, std::int64_t incx, float beta, float *y,
                     std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    gbmv_precondition(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                      incy, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::gbmv(selector.get_queue(), trans, m, n, kl, ku, alpha, a,
                                               lda, x, incx, beta, y, incy, dependencies);
    gbmv_postcondition(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                       incy, dependencies);
    return done;
}

sycl::event gbmv(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, double alpha,
                     const double *a, std::int64_t lda, const double *x, std::int64_t incx,
                     double beta, double *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    gbmv_precondition(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                      incy, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::gbmv(selector.get_queue(), trans, m, n, kl, ku, alpha, a,
                                               lda, x, incx, beta, y, incy, dependencies);
    gbmv_postcondition(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                       incy, dependencies);
    return done;
}

sycl::event gbmv(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *x,
                     std::int64_t incx, std::complex<float> beta, std::complex<float> *y,
                     std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    gbmv_precondition(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                      incy, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::gbmv(selector.get_queue(), trans, m, n, kl, ku, alpha, a,
                                               lda, x, incx, beta, y, incy, dependencies);
    gbmv_postcondition(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                       incy, dependencies);
    return done;
}

sycl::event gbmv(backend_selector<backend::mklgpu> selector, transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *x,
                     std::int64_t incx, std::complex<double> beta, std::complex<double> *y,
                     std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    gbmv_precondition(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                      incy, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::gbmv(selector.get_queue(), trans, m, n, kl, ku, alpha, a,
                                               lda, x, incx, beta, y, incy, dependencies);
    gbmv_postcondition(selector.get_queue(), trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                       incy, dependencies);
    return done;
}

sycl::event tbmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, std::int64_t k, const float *a,
                     std::int64_t lda, float *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    tbmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::tbmv(
        selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    tbmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx,
                       dependencies);
    return done;
}

sycl::event tbmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, std::int64_t k, const double *a,
                     std::int64_t lda, double *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    tbmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::tbmv(
        selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    tbmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx,
                       dependencies);
    return done;
}

sycl::event tbmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, std::int64_t k, const std::complex<float> *a,
                     std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    tbmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::tbmv(
        selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    tbmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx,
                       dependencies);
    return done;
}

sycl::event tbmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, std::int64_t k, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    tbmv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::tbmv(
        selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    tbmv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx,
                       dependencies);
    return done;
}

sycl::event symm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
                     std::int64_t m, std::int64_t n, float alpha, const float *a, std::int64_t lda,
                     const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    symm_precondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                      beta, c, ldc, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::symm(selector.get_queue(), left_right, upper_lower, m, n,
                                               alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
    symm_postcondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                       beta, c, ldc, dependencies);
    return done;
}

sycl::event symm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
                     std::int64_t m, std::int64_t n, double alpha, const double *a,
                     std::int64_t lda, const double *b, std::int64_t ldb, double beta, double *c,
                     std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    symm_precondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                      beta, c, ldc, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::symm(selector.get_queue(), left_right, upper_lower, m, n,
                                               alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
    symm_postcondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                       beta, c, ldc, dependencies);
    return done;
}

sycl::event symm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
                     std::int64_t m, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                     std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                     std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    symm_precondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                      beta, c, ldc, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::symm(selector.get_queue(), left_right, upper_lower, m, n,
                                               alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
    symm_postcondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                       beta, c, ldc, dependencies);
    return done;
}

sycl::event symm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
                     std::int64_t m, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
                     std::int64_t ldb, std::complex<double> beta, std::complex<double> *c,
                     std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    symm_precondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                      beta, c, ldc, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::symm(selector.get_queue(), left_right, upper_lower, m, n,
                                               alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
    symm_postcondition(selector.get_queue(), left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                       beta, c, ldc, dependencies);
    return done;
}

sycl::event dotc(backend_selector<backend::mklgpu> selector, std::int64_t n,
                     const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
                     std::int64_t incy, std::complex<float> *result,
                     const std::vector<sycl::event> &dependencies) {
    dotc_precondition(selector.get_queue(), n, x, incx, y, incy, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::dotc(selector.get_queue(), n, x, incx, y, incy,
                                                       result, dependencies);
    dotc_postcondition(selector.get_queue(), n, x, incx, y, incy, result, dependencies);
    return done;
}

sycl::event dotc(backend_selector<backend::mklgpu> selector, std::int64_t n,
                     const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *result,
                     const std::vector<sycl::event> &dependencies) {
    dotc_precondition(selector.get_queue(), n, x, incx, y, incy, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::dotc(selector.get_queue(), n, x, incx, y, incy,
                                                       result, dependencies);
    dotc_postcondition(selector.get_queue(), n, x, incx, y, incy, result, dependencies);
    return done;
}

sycl::event syr(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                    float alpha, const float *x, std::int64_t incx, float *a, std::int64_t lda,
                    const std::vector<sycl::event> &dependencies) {
    syr_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, lda, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::syr(selector.get_queue(), upper_lower, n, alpha,
                                                      x, incx, a, lda, dependencies);
    syr_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, lda, dependencies);
    return done;
}

sycl::event syr(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                    double alpha, const double *x, std::int64_t incx, double *a, std::int64_t lda,
                    const std::vector<sycl::event> &dependencies) {
    syr_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, lda, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::syr(selector.get_queue(), upper_lower, n, alpha,
                                                      x, incx, a, lda, dependencies);
    syr_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, a, lda, dependencies);
    return done;
}

sycl::event trmm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                     const float *a, std::int64_t lda, float *b, std::int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    trmm_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                      a, lda, b, ldb, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trmm(selector.get_queue(), left_right,
                                                       upper_lower, trans, unit_diag, m, n, alpha,
                                                       a, lda, b, ldb, dependencies);
    trmm_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                       a, lda, b, ldb, dependencies);
    return done;
}

sycl::event trmm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                     const double *a, std::int64_t lda, double *b, std::int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    trmm_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                      a, lda, b, ldb, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trmm(selector.get_queue(), left_right,
                                                       upper_lower, trans, unit_diag, m, n, alpha,
                                                       a, lda, b, ldb, dependencies);
    trmm_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                       a, lda, b, ldb, dependencies);
    return done;
}

sycl::event trmm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> *b, std::int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    trmm_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                      a, lda, b, ldb, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trmm(selector.get_queue(), left_right,
                                                       upper_lower, trans, unit_diag, m, n, alpha,
                                                       a, lda, b, ldb, dependencies);
    trmm_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                       a, lda, b, ldb, dependencies);
    return done;
}

sycl::event trmm(backend_selector<backend::mklgpu> selector, side left_right, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     std::complex<double> *b, std::int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    trmm_precondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                      a, lda, b, ldb, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trmm(selector.get_queue(), left_right,
                                                       upper_lower, trans, unit_diag, m, n, alpha,
                                                       a, lda, b, ldb, dependencies);
    trmm_postcondition(selector.get_queue(), left_right, upper_lower, trans, unit_diag, m, n, alpha,
                       a, lda, b, ldb, dependencies);
    return done;
}

sycl::event rotmg(backend_selector<backend::mklgpu> selector, float *d1, float *d2, float *x1,
                      float y1, float *param, const std::vector<sycl::event> &dependencies) {
    rotmg_precondition(selector.get_queue(), d1, d2, x1, y1, param, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::rotmg(selector.get_queue(), d1, d2, x1, y1, param,
                                                        dependencies);
    rotmg_postcondition(selector.get_queue(), d1, d2, x1, y1, param, dependencies);
    return done;
}

sycl::event rotmg(backend_selector<backend::mklgpu> selector, double *d1, double *d2,
                      double *x1, double y1, double *param,
                      const std::vector<sycl::event> &dependencies) {
    rotmg_precondition(selector.get_queue(), d1, d2, x1, y1, param, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::rotmg(selector.get_queue(), d1, d2, x1, y1, param,
                                                        dependencies);
    rotmg_postcondition(selector.get_queue(), d1, d2, x1, y1, param, dependencies);
    return done;
}

sycl::event tpsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, const float *a, float *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    tpsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::tpsv(selector.get_queue(), upper_lower, trans,
                                                       unit_diag, n, a, x, incx, dependencies);
    tpsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx,
                       dependencies);
    return done;
}

sycl::event tpsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, const double *a, double *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    tpsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::tpsv(selector.get_queue(), upper_lower, trans,
                                                       unit_diag, n, a, x, incx, dependencies);
    tpsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx,
                       dependencies);
    return done;
}

sycl::event tpsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, const std::complex<float> *a,
                     std::complex<float> *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    tpsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::tpsv(selector.get_queue(), upper_lower, trans,
                                                       unit_diag, n, a, x, incx, dependencies);
    tpsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx,
                       dependencies);
    return done;
}

sycl::event tpsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, const std::complex<double> *a,
                     std::complex<double> *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    tpsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::tpsv(selector.get_queue(), upper_lower, trans,
                                                       unit_diag, n, a, x, incx, dependencies);
    tpsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, x, incx,
                       dependencies);
    return done;
}

sycl::event trsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, const float *a, std::int64_t lda, float *x,
                     std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    trsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trsv(selector.get_queue(), upper_lower, trans,
                                                       unit_diag, n, a, lda, x, incx, dependencies);
    trsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx,
                       dependencies);
    return done;
}

sycl::event trsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, const double *a, std::int64_t lda, double *x,
                     std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    trsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trsv(selector.get_queue(), upper_lower, trans,
                                                       unit_diag, n, a, lda, x, incx, dependencies);
    trsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx,
                       dependencies);
    return done;
}

sycl::event trsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    trsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trsv(selector.get_queue(), upper_lower, trans,
                                                       unit_diag, n, a, lda, x, incx, dependencies);
    trsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx,
                       dependencies);
    return done;
}

sycl::event trsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    trsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::trsv(selector.get_queue(), upper_lower, trans,
                                                       unit_diag, n, a, lda, x, incx, dependencies);
    trsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, a, lda, x, incx,
                       dependencies);
    return done;
}

sycl::event copy(backend_selector<backend::mklgpu> selector, std::int64_t n, const float *x,
                     std::int64_t incx, float *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    copy_precondition(selector.get_queue(), n, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::copy(selector.get_queue(), n, x, incx, y, incy,
                                                       dependencies);
    copy_postcondition(selector.get_queue(), n, x, incx, y, incy, dependencies);
    return done;
}

sycl::event copy(backend_selector<backend::mklgpu> selector, std::int64_t n, const double *x,
                     std::int64_t incx, double *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    copy_precondition(selector.get_queue(), n, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::copy(selector.get_queue(), n, x, incx, y, incy,
                                                       dependencies);
    copy_postcondition(selector.get_queue(), n, x, incx, y, incy, dependencies);
    return done;
}

sycl::event copy(backend_selector<backend::mklgpu> selector, std::int64_t n,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> *y,
                     std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    copy_precondition(selector.get_queue(), n, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::copy(selector.get_queue(), n, x, incx, y, incy,
                                                       dependencies);
    copy_postcondition(selector.get_queue(), n, x, incx, y, incy, dependencies);
    return done;
}

sycl::event copy(backend_selector<backend::mklgpu> selector, std::int64_t n,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> *y,
                     std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    copy_precondition(selector.get_queue(), n, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::copy(selector.get_queue(), n, x, incx, y, incy,
                                                       dependencies);
    copy_postcondition(selector.get_queue(), n, x, incx, y, incy, dependencies);
    return done;
}

sycl::event copy_batch(backend_selector<backend::mklgpu> selector, std::int64_t *n,
                           const float **x, std::int64_t *incx, float **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    copy_batch_precondition(selector.get_queue(), n, x, incx, y, incy, group_count, group_size,
                            dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::copy_batch(
        selector.get_queue(), n, x, incx, y, incy, group_count, group_size, dependencies);
    copy_batch_postcondition(selector.get_queue(), n, x, incx, y, incy, group_count, group_size,
                             dependencies);
    return done;
}

sycl::event copy_batch(backend_selector<backend::mklgpu> selector, std::int64_t *n,
                           const double **x, std::int64_t *incx, double **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    copy_batch_precondition(selector.get_queue(), n, x, incx, y, incy, group_count, group_size,
                            dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::copy_batch(
        selector.get_queue(), n, x, incx, y, incy, group_count, group_size, dependencies);
    copy_batch_postcondition(selector.get_queue(), n, x, incx, y, incy, group_count, group_size,
                             dependencies);
    return done;
}

sycl::event copy_batch(backend_selector<backend::mklgpu> selector, std::int64_t *n,
                           const std::complex<float> **x, std::int64_t *incx,
                           std::complex<float> **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    copy_batch_precondition(selector.get_queue(), n, x, incx, y, incy, group_count, group_size,
                            dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::copy_batch(
        selector.get_queue(), n, x, incx, y, incy, group_count, group_size, dependencies);
    copy_batch_postcondition(selector.get_queue(), n, x, incx, y, incy, group_count, group_size,
                             dependencies);
    return done;
}

sycl::event copy_batch(backend_selector<backend::mklgpu> selector, std::int64_t *n,
                           const std::complex<double> **x, std::int64_t *incx,
                           std::complex<double> **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    copy_batch_precondition(selector.get_queue(), n, x, incx, y, incy, group_count, group_size,
                            dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::copy_batch(
        selector.get_queue(), n, x, incx, y, incy, group_count, group_size, dependencies);
    copy_batch_postcondition(selector.get_queue(), n, x, incx, y, incy, group_count, group_size,
                             dependencies);
    return done;
}

sycl::event copy_batch(backend_selector<backend::mklgpu> selector, std::int64_t n,
                           const float *x, std::int64_t incx, std::int64_t stridex, float *y,
                           std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    copy_batch_precondition(selector.get_queue(), n, x, incx, stridex, y, incy, stridey, batch_size,
                            dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::copy_batch(
        selector.get_queue(), n, x, incx, stridex, y, incy, stridey, batch_size, dependencies);
    copy_batch_postcondition(selector.get_queue(), n, x, incx, stridex, y, incy, stridey,
                             batch_size, dependencies);
    return done;
}

sycl::event copy_batch(backend_selector<backend::mklgpu> selector, std::int64_t n,
                           const double *x, std::int64_t incx, std::int64_t stridex, double *y,
                           std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    copy_batch_precondition(selector.get_queue(), n, x, incx, stridex, y, incy, stridey, batch_size,
                            dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::copy_batch(
        selector.get_queue(), n, x, incx, stridex, y, incy, stridey, batch_size, dependencies);
    copy_batch_postcondition(selector.get_queue(), n, x, incx, stridex, y, incy, stridey,
                             batch_size, dependencies);
    return done;
}

sycl::event copy_batch(backend_selector<backend::mklgpu> selector, std::int64_t n,
                           const std::complex<float> *x, std::int64_t incx, std::int64_t stridex,
                           std::complex<float> *y, std::int64_t incy, std::int64_t stridey,
                           std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    copy_batch_precondition(selector.get_queue(), n, x, incx, stridex, y, incy, stridey, batch_size,
                            dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::copy_batch(
        selector.get_queue(), n, x, incx, stridex, y, incy, stridey, batch_size, dependencies);
    copy_batch_postcondition(selector.get_queue(), n, x, incx, stridex, y, incy, stridey,
                             batch_size, dependencies);
    return done;
}

sycl::event copy_batch(backend_selector<backend::mklgpu> selector, std::int64_t n,
                           const std::complex<double> *x, std::int64_t incx, std::int64_t stridex,
                           std::complex<double> *y, std::int64_t incy, std::int64_t stridey,
                           std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    copy_batch_precondition(selector.get_queue(), n, x, incx, stridex, y, incy, stridey, batch_size,
                            dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::copy_batch(
        selector.get_queue(), n, x, incx, stridex, y, incy, stridey, batch_size, dependencies);
    copy_batch_postcondition(selector.get_queue(), n, x, incx, stridex, y, incy, stridey,
                             batch_size, dependencies);
    return done;
}

sycl::event hemv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    hemv_precondition(selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::hemv(
        selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    hemv_postcondition(selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

sycl::event hemv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    hemv_precondition(selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::hemv(
        selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    hemv_postcondition(selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

sycl::event gemmt(backend_selector<backend::mklgpu> selector, uplo upper_lower,
                      transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                      float alpha, const float *a, std::int64_t lda, const float *b,
                      std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                      const std::vector<sycl::event> &dependencies) {
    gemmt_precondition(selector.get_queue(), upper_lower, transa, transb, n, k, alpha, a, lda, b,
                       ldb, beta, c, ldc, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemmt(selector.get_queue(), upper_lower, transa,
                                                        transb, n, k, alpha, a, lda, b, ldb, beta,
                                                        c, ldc, dependencies);
    gemmt_postcondition(selector.get_queue(), upper_lower, transa, transb, n, k, alpha, a, lda, b,
                        ldb, beta, c, ldc, dependencies);
    return done;
}

sycl::event gemmt(backend_selector<backend::mklgpu> selector, uplo upper_lower,
                      transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                      double alpha, const double *a, std::int64_t lda, const double *b,
                      std::int64_t ldb, double beta, double *c, std::int64_t ldc,
                      const std::vector<sycl::event> &dependencies) {
    gemmt_precondition(selector.get_queue(), upper_lower, transa, transb, n, k, alpha, a, lda, b,
                       ldb, beta, c, ldc, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemmt(selector.get_queue(), upper_lower, transa,
                                                        transb, n, k, alpha, a, lda, b, ldb, beta,
                                                        c, ldc, dependencies);
    gemmt_postcondition(selector.get_queue(), upper_lower, transa, transb, n, k, alpha, a, lda, b,
                        ldb, beta, c, ldc, dependencies);
    return done;
}

sycl::event gemmt(backend_selector<backend::mklgpu> selector, uplo upper_lower,
                      transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                      std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                      const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
                      std::complex<float> *c, std::int64_t ldc,
                      const std::vector<sycl::event> &dependencies) {
    gemmt_precondition(selector.get_queue(), upper_lower, transa, transb, n, k, alpha, a, lda, b,
                       ldb, beta, c, ldc, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemmt(selector.get_queue(), upper_lower, transa,
                                                        transb, n, k, alpha, a, lda, b, ldb, beta,
                                                        c, ldc, dependencies);
    gemmt_postcondition(selector.get_queue(), upper_lower, transa, transb, n, k, alpha, a, lda, b,
                        ldb, beta, c, ldc, dependencies);
    return done;
}

sycl::event gemmt(backend_selector<backend::mklgpu> selector, uplo upper_lower,
                      transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                      std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                      const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                      std::complex<double> *c, std::int64_t ldc,
                      const std::vector<sycl::event> &dependencies) {
    gemmt_precondition(selector.get_queue(), upper_lower, transa, transb, n, k, alpha, a, lda, b,
                       ldb, beta, c, ldc, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::gemmt(selector.get_queue(), upper_lower, transa,
                                                        transb, n, k, alpha, a, lda, b, ldb, beta,
                                                        c, ldc, dependencies);
    gemmt_postcondition(selector.get_queue(), upper_lower, transa, transb, n, k, alpha, a, lda, b,
                        ldb, beta, c, ldc, dependencies);
    return done;
}

sycl::event sbmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     std::int64_t k, float alpha, const float *a, std::int64_t lda, const float *x,
                     std::int64_t incx, float beta, float *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    sbmv_precondition(selector.get_queue(), upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                      incy, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::sbmv(selector.get_queue(), upper_lower, n, k, alpha, a,
                                               lda, x, incx, beta, y, incy, dependencies);
    sbmv_postcondition(selector.get_queue(), upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                       incy, dependencies);
    return done;
}

sycl::event sbmv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     std::int64_t k, double alpha, const double *a, std::int64_t lda,
                     const double *x, std::int64_t incx, double beta, double *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    sbmv_precondition(selector.get_queue(), upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                      incy, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::sbmv(selector.get_queue(), upper_lower, n, k, alpha, a,
                                               lda, x, incx, beta, y, incy, dependencies);
    sbmv_postcondition(selector.get_queue(), upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                       incy, dependencies);
    return done;
}

sycl::event asum(backend_selector<backend::mklgpu> selector, std::int64_t n,
                     const std::complex<float> *x, std::int64_t incx, float *result,
                     const std::vector<sycl::event> &dependencies) {
    asum_precondition(selector.get_queue(), n, x, incx, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::asum(selector.get_queue(), n, x, incx, result,
                                                       dependencies);
    asum_postcondition(selector.get_queue(), n, x, incx, result, dependencies);
    return done;
}

sycl::event asum(backend_selector<backend::mklgpu> selector, std::int64_t n,
                     const std::complex<double> *x, std::int64_t incx, double *result,
                     const std::vector<sycl::event> &dependencies) {
    asum_precondition(selector.get_queue(), n, x, incx, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::asum(selector.get_queue(), n, x, incx, result,
                                                       dependencies);
    asum_postcondition(selector.get_queue(), n, x, incx, result, dependencies);
    return done;
}

sycl::event asum(backend_selector<backend::mklgpu> selector, std::int64_t n, const float *x,
                     std::int64_t incx, float *result,
                     const std::vector<sycl::event> &dependencies) {
    asum_precondition(selector.get_queue(), n, x, incx, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::asum(selector.get_queue(), n, x, incx, result,
                                                       dependencies);
    asum_postcondition(selector.get_queue(), n, x, incx, result, dependencies);
    return done;
}

sycl::event asum(backend_selector<backend::mklgpu> selector, std::int64_t n, const double *x,
                     std::int64_t incx, double *result,
                     const std::vector<sycl::event> &dependencies) {
    asum_precondition(selector.get_queue(), n, x, incx, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::asum(selector.get_queue(), n, x, incx, result,
                                                       dependencies);
    asum_postcondition(selector.get_queue(), n, x, incx, result, dependencies);
    return done;
}

sycl::event tbsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, std::int64_t k, const float *a,
                     std::int64_t lda, float *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    tbsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::tbsv(
        selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    tbsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx,
                       dependencies);
    return done;
}

sycl::event tbsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, std::int64_t k, const double *a,
                     std::int64_t lda, double *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    tbsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::tbsv(
        selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    tbsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx,
                       dependencies);
    return done;
}

sycl::event tbsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, std::int64_t k, const std::complex<float> *a,
                     std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    tbsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::tbsv(
        selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    tbsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx,
                       dependencies);
    return done;
}

sycl::event tbsv(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t n, std::int64_t k, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    tbsv_precondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::tbsv(
        selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    tbsv_postcondition(selector.get_queue(), upper_lower, trans, unit_diag, n, k, a, lda, x, incx,
                       dependencies);
    return done;
}

sycl::event spr2(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     float alpha, const float *x, std::int64_t incx, const float *y,
                     std::int64_t incy, float *a,
                     const std::vector<sycl::event> &dependencies) {
    spr2_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::spr2(selector.get_queue(), upper_lower, n, alpha,
                                                       x, incx, y, incy, a, dependencies);
    spr2_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a,
                       dependencies);
    return done;
}

sycl::event spr2(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     double alpha, const double *x, std::int64_t incx, const double *y,
                     std::int64_t incy, double *a,
                     const std::vector<sycl::event> &dependencies) {
    spr2_precondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::spr2(selector.get_queue(), upper_lower, n, alpha,
                                                       x, incx, y, incy, a, dependencies);
    spr2_postcondition(selector.get_queue(), upper_lower, n, alpha, x, incx, y, incy, a,
                       dependencies);
    return done;
}

sycl::event iamax(backend_selector<backend::mklgpu> selector, std::int64_t n, const float *x,
                      std::int64_t incx, std::int64_t *result,
                      const std::vector<sycl::event> &dependencies) {
    iamax_precondition(selector.get_queue(), n, x, incx, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::iamax(selector.get_queue(), n, x, incx, result,
                                                        dependencies);
    iamax_postcondition(selector.get_queue(), n, x, incx, result, dependencies);
    return done;
}

sycl::event iamax(backend_selector<backend::mklgpu> selector, std::int64_t n, const double *x,
                      std::int64_t incx, std::int64_t *result,
                      const std::vector<sycl::event> &dependencies) {
    iamax_precondition(selector.get_queue(), n, x, incx, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::iamax(selector.get_queue(), n, x, incx, result,
                                                        dependencies);
    iamax_postcondition(selector.get_queue(), n, x, incx, result, dependencies);
    return done;
}

sycl::event iamax(backend_selector<backend::mklgpu> selector, std::int64_t n,
                      const std::complex<float> *x, std::int64_t incx, std::int64_t *result,
                      const std::vector<sycl::event> &dependencies) {
    iamax_precondition(selector.get_queue(), n, x, incx, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::iamax(selector.get_queue(), n, x, incx, result,
                                                        dependencies);
    iamax_postcondition(selector.get_queue(), n, x, incx, result, dependencies);
    return done;
}

sycl::event iamax(backend_selector<backend::mklgpu> selector, std::int64_t n,
                      const std::complex<double> *x, std::int64_t incx, std::int64_t *result,
                      const std::vector<sycl::event> &dependencies) {
    iamax_precondition(selector.get_queue(), n, x, incx, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::iamax(selector.get_queue(), n, x, incx, result,
                                                        dependencies);
    iamax_postcondition(selector.get_queue(), n, x, incx, result, dependencies);
    return done;
}

sycl::event rotm(backend_selector<backend::mklgpu> selector, std::int64_t n, float *x,
                     std::int64_t incx, float *y, std::int64_t incy, float *param,
                     const std::vector<sycl::event> &dependencies) {
    rotm_precondition(selector.get_queue(), n, x, incx, y, incy, param, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::rotm(selector.get_queue(), n, x, incx, y, incy,
                                                       param, dependencies);
    rotm_postcondition(selector.get_queue(), n, x, incx, y, incy, param, dependencies);
    return done;
}

sycl::event rotm(backend_selector<backend::mklgpu> selector, std::int64_t n, double *x,
                     std::int64_t incx, double *y, std::int64_t incy, double *param,
                     const std::vector<sycl::event> &dependencies) {
    rotm_precondition(selector.get_queue(), n, x, incx, y, incy, param, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::rotm(selector.get_queue(), n, x, incx, y, incy,
                                                       param, dependencies);
    rotm_postcondition(selector.get_queue(), n, x, incx, y, incy, param, dependencies);
    return done;
}

sycl::event rotg(backend_selector<backend::mklgpu> selector, float *a, float *b, float *c,
                     float *s, const std::vector<sycl::event> &dependencies) {
    rotg_precondition(selector.get_queue(), a, b, c, s, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::rotg(selector.get_queue(), a, b, c, s, dependencies);
    rotg_postcondition(selector.get_queue(), a, b, c, s, dependencies);
    return done;
}

sycl::event rotg(backend_selector<backend::mklgpu> selector, double *a, double *b, double *c,
                     double *s, const std::vector<sycl::event> &dependencies) {
    rotg_precondition(selector.get_queue(), a, b, c, s, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::rotg(selector.get_queue(), a, b, c, s, dependencies);
    rotg_postcondition(selector.get_queue(), a, b, c, s, dependencies);
    return done;
}

sycl::event rotg(backend_selector<backend::mklgpu> selector, std::complex<float> *a,
                     std::complex<float> *b, float *c, std::complex<float> *s,
                     const std::vector<sycl::event> &dependencies) {
    rotg_precondition(selector.get_queue(), a, b, c, s, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::rotg(selector.get_queue(), a, b, c, s, dependencies);
    rotg_postcondition(selector.get_queue(), a, b, c, s, dependencies);
    return done;
}

sycl::event rotg(backend_selector<backend::mklgpu> selector, std::complex<double> *a,
                     std::complex<double> *b, double *c, std::complex<double> *s,
                     const std::vector<sycl::event> &dependencies) {
    rotg_precondition(selector.get_queue(), a, b, c, s, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::rotg(selector.get_queue(), a, b, c, s, dependencies);
    rotg_postcondition(selector.get_queue(), a, b, c, s, dependencies);
    return done;
}

sycl::event sdsdot(backend_selector<backend::mklgpu> selector, std::int64_t n, float sb,
                       const float *x, std::int64_t incx, const float *y, std::int64_t incy,
                       float *result, const std::vector<sycl::event> &dependencies) {
    sdsdot_precondition(selector.get_queue(), n, sb, x, incx, y, incy, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::sdsdot(selector.get_queue(), n, sb, x, incx, y,
                                                         incy, result, dependencies);
    sdsdot_postcondition(selector.get_queue(), n, sb, x, incx, y, incy, result, dependencies);
    return done;
}

sycl::event her2k(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                      std::int64_t n, std::int64_t k, std::complex<float> alpha,
                      const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                      std::int64_t ldb, float beta, std::complex<float> *c, std::int64_t ldc,
                      const std::vector<sycl::event> &dependencies) {
    her2k_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::her2k(selector.get_queue(), upper_lower, trans, n, k,
                                                alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
    her2k_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                        c, ldc, dependencies);
    return done;
}

sycl::event her2k(backend_selector<backend::mklgpu> selector, uplo upper_lower, transpose trans,
                      std::int64_t n, std::int64_t k, std::complex<double> alpha,
                      const std::complex<double> *a, std::int64_t lda,
                      const std::complex<double> *b, std::int64_t ldb, double beta,
                      std::complex<double> *c, std::int64_t ldc,
                      const std::vector<sycl::event> &dependencies) {
    her2k_precondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc, dependencies);
    auto done =
        oneapi::mkl::blas::mklgpu::MAJOR::her2k(selector.get_queue(), upper_lower, trans, n, k,
                                                alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
    her2k_postcondition(selector.get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                        c, ldc, dependencies);
    return done;
}

sycl::event dot(backend_selector<backend::mklgpu> selector, std::int64_t n, const float *x,
                    std::int64_t incx, const float *y, std::int64_t incy, float *result,
                    const std::vector<sycl::event> &dependencies) {
    dot_precondition(selector.get_queue(), n, x, incx, y, incy, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::dot(selector.get_queue(), n, x, incx, y, incy,
                                                      result, dependencies);
    dot_postcondition(selector.get_queue(), n, x, incx, y, incy, result, dependencies);
    return done;
}

sycl::event dot(backend_selector<backend::mklgpu> selector, std::int64_t n, const double *x,
                    std::int64_t incx, const double *y, std::int64_t incy, double *result,
                    const std::vector<sycl::event> &dependencies) {
    dot_precondition(selector.get_queue(), n, x, incx, y, incy, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::dot(selector.get_queue(), n, x, incx, y, incy,
                                                      result, dependencies);
    dot_postcondition(selector.get_queue(), n, x, incx, y, incy, result, dependencies);
    return done;
}

sycl::event dot(backend_selector<backend::mklgpu> selector, std::int64_t n, const float *x,
                    std::int64_t incx, const float *y, std::int64_t incy, double *result,
                    const std::vector<sycl::event> &dependencies) {
    dot_precondition(selector.get_queue(), n, x, incx, y, incy, result, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::dot(selector.get_queue(), n, x, incx, y, incy,
                                                      result, dependencies);
    dot_postcondition(selector.get_queue(), n, x, incx, y, incy, result, dependencies);
    return done;
}

sycl::event symv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     float alpha, const float *a, std::int64_t lda, const float *x,
                     std::int64_t incx, float beta, float *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    symv_precondition(selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::symv(
        selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    symv_postcondition(selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

sycl::event symv(backend_selector<backend::mklgpu> selector, uplo upper_lower, std::int64_t n,
                     double alpha, const double *a, std::int64_t lda, const double *x,
                     std::int64_t incx, double beta, double *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    symv_precondition(selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::symv(
        selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    symv_postcondition(selector.get_queue(), upper_lower, n, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

sycl::event omatcopy_batch(backend_selector<backend::mklgpu> selector, transpose trans,
                           std::int64_t m, std::int64_t n, float alpha, const float *a,
                           std::int64_t lda, std::int64_t stride_a, float *b, std::int64_t ldb,
                           std::int64_t stride_b, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    omatcopy_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb,
                                stride_b, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::omatcopy_batch(
        selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb, stride_b, batch_size,
        dependencies);
    omatcopy_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb,
                                 stride_b, batch_size, dependencies);
    return done;
}

sycl::event omatcopy_batch(backend_selector<backend::mklgpu> selector, transpose trans,
                           std::int64_t m, std::int64_t n, double alpha, const double *a,
                           std::int64_t lda, std::int64_t stride_a, double *b, std::int64_t ldb,
                           std::int64_t stride_b, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    omatcopy_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb,
                                stride_b, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::omatcopy_batch(
        selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb, stride_b, batch_size,
        dependencies);
    omatcopy_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb,
                                 stride_b, batch_size, dependencies);
    return done;
}

sycl::event omatcopy_batch(backend_selector<backend::mklgpu> selector, transpose trans,
                           std::int64_t m, std::int64_t n, std::complex<float> alpha,
                           const std::complex<float> *a, std::int64_t lda, std::int64_t stride_a,
                           std::complex<float> *b, std::int64_t ldb, std::int64_t stride_b,
                           std::int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    omatcopy_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb,
                                stride_b, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::omatcopy_batch(
        selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb, stride_b, batch_size,
        dependencies);
    omatcopy_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb,
                                 stride_b, batch_size, dependencies);
    return done;
}

sycl::event omatcopy_batch(backend_selector<backend::mklgpu> selector, transpose trans,
                           std::int64_t m, std::int64_t n, std::complex<double> alpha,
                           const std::complex<double> *a, std::int64_t lda, std::int64_t stride_a,
                           std::complex<double> *b, std::int64_t ldb, std::int64_t stride_b,
                           std::int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    omatcopy_batch_precondition(selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb,
                                stride_b, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::omatcopy_batch(
        selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb, stride_b, batch_size,
        dependencies);
    omatcopy_batch_postcondition(selector.get_queue(), trans, m, n, alpha, a, lda, stride_a, b, ldb,
                                 stride_b, batch_size, dependencies);
    return done;
}

sycl::event imatcopy_batch(backend_selector<backend::mklgpu> selector, transpose trans,
                           std::int64_t m, std::int64_t n, float alpha, float *ab, std::int64_t lda,
                           std::int64_t ldb, std::int64_t stride, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    imatcopy_batch_precondition(selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride,
                                batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::imatcopy_batch(
        selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride, batch_size, dependencies);
    imatcopy_batch_postcondition(selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride,
                                 batch_size, dependencies);
    return done;
}

sycl::event imatcopy_batch(backend_selector<backend::mklgpu> selector, transpose trans,
                           std::int64_t m, std::int64_t n, double alpha, double *ab,
                           std::int64_t lda, std::int64_t ldb, std::int64_t stride,
                           std::int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    imatcopy_batch_precondition(selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride,
                                batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::imatcopy_batch(
        selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride, batch_size, dependencies);
    imatcopy_batch_postcondition(selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride,
                                 batch_size, dependencies);
    return done;
}

sycl::event imatcopy_batch(backend_selector<backend::mklgpu> selector, transpose trans,
                           std::int64_t m, std::int64_t n, std::complex<float> alpha,
                           std::complex<float> *ab, std::int64_t lda, std::int64_t ldb,
                           std::int64_t stride, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    imatcopy_batch_precondition(selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride,
                                batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::imatcopy_batch(
        selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride, batch_size, dependencies);
    imatcopy_batch_postcondition(selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride,
                                 batch_size, dependencies);
    return done;
}

sycl::event imatcopy_batch(backend_selector<backend::mklgpu> selector, transpose trans,
                           std::int64_t m, std::int64_t n, std::complex<double> alpha,
                           std::complex<double> *ab, std::int64_t lda, std::int64_t ldb,
                           std::int64_t stride, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    imatcopy_batch_precondition(selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride,
                                batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::imatcopy_batch(
        selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride, batch_size, dependencies);
    imatcopy_batch_postcondition(selector.get_queue(), trans, m, n, alpha, ab, lda, ldb, stride,
                                 batch_size, dependencies);
    return done;
}

sycl::event omatadd_batch(backend_selector<backend::mklgpu> selector, transpose transa,
                          transpose transb, std::int64_t m, std::int64_t n, float alpha,
                          const float *a, std::int64_t lda, std::int64_t stride_a, float beta,
                          const float *b, std::int64_t ldb, std::int64_t stride_b, float *c,
                          std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                          const std::vector<sycl::event> &dependencies) {
    omatadd_batch_precondition(selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a,
                               beta, b, ldb, stride_b, c, ldc, stride_c, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::omatadd_batch(
        selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a, beta, b, ldb, stride_b,
        c, ldc, stride_c, batch_size, dependencies);
    omatadd_batch_postcondition(selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a,
                                beta, b, ldb, stride_b, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

sycl::event omatadd_batch(backend_selector<backend::mklgpu> selector, transpose transa,
                          transpose transb, std::int64_t m, std::int64_t n, double alpha,
                          const double *a, std::int64_t lda, std::int64_t stride_a, double beta,
                          const double *b, std::int64_t ldb, std::int64_t stride_b, double *c,
                          std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                          const std::vector<sycl::event> &dependencies) {
    omatadd_batch_precondition(selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a,
                               beta, b, ldb, stride_b, c, ldc, stride_c, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::omatadd_batch(
        selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a, beta, b, ldb, stride_b,
        c, ldc, stride_c, batch_size, dependencies);
    omatadd_batch_postcondition(selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a,
                                beta, b, ldb, stride_b, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

sycl::event omatadd_batch(backend_selector<backend::mklgpu> selector, transpose transa,
                          transpose transb, std::int64_t m, std::int64_t n,
                          std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                          std::int64_t stride_a, std::complex<float> beta,
                          const std::complex<float> *b, std::int64_t ldb, std::int64_t stride_b,
                          std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
                          std::int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    omatadd_batch_precondition(selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a,
                               beta, b, ldb, stride_b, c, ldc, stride_c, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::omatadd_batch(
        selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a, beta, b, ldb, stride_b,
        c, ldc, stride_c, batch_size, dependencies);
    omatadd_batch_postcondition(selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a,
                                beta, b, ldb, stride_b, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

sycl::event omatadd_batch(backend_selector<backend::mklgpu> selector, transpose transa,
                          transpose transb, std::int64_t m, std::int64_t n,
                          std::complex<double> alpha, const std::complex<double> *a,
                          std::int64_t lda, std::int64_t stride_a, std::complex<double> beta,
                          const std::complex<double> *b, std::int64_t ldb, std::int64_t stride_b,
                          std::complex<double> *c, std::int64_t ldc, std::int64_t stride_c,
                          std::int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    omatadd_batch_precondition(selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a,
                               beta, b, ldb, stride_b, c, ldc, stride_c, batch_size, dependencies);
    auto done = oneapi::mkl::blas::mklgpu::MAJOR::omatadd_batch(
        selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a, beta, b, ldb, stride_b,
        c, ldc, stride_c, batch_size, dependencies);
    omatadd_batch_postcondition(selector.get_queue(), transa, transb, m, n, alpha, a, lda, stride_a,
                                beta, b, ldb, stride_b, c, ldc, stride_c, batch_size, dependencies);
    return done;
}
