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

static inline void syr2(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda);

static inline void syr2(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda);

static inline void scal(backend_selector<backend::BACKEND> selector, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx);

static inline void scal(backend_selector<backend::BACKEND> selector, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx);

static inline void scal(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx);

static inline void scal(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx);

static inline void scal(backend_selector<backend::BACKEND> selector, std::int64_t n, float alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);

static inline void scal(backend_selector<backend::BACKEND> selector, std::int64_t n, double alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

static inline void trmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx);

static inline void trmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx);

static inline void trmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);

static inline void trmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

static inline void tpmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx);

static inline void tpmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx);

static inline void tpmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);

static inline void tpmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

static inline void spr(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                       std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &a);

static inline void spr(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                       std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<double, 1> &a);

static inline void hpmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy);

static inline void hpmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy);

static inline void syrk(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, std::int64_t n, std::int64_t k, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda, float beta,
                        cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

static inline void syrk(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, std::int64_t n, std::int64_t k, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda, double beta,
                        cl::sycl::buffer<double, 1> &c, std::int64_t ldc);

static inline void syrk(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc);

static inline void syrk(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, std::int64_t n, std::int64_t k, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc);

static inline void her2(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);

static inline void her2(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);

static inline void hbmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, std::int64_t k, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy);

static inline void hbmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, std::int64_t k, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy);

static inline void rot(backend_selector<backend::BACKEND> selector, std::int64_t n,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy, float c,
                       float s);

static inline void rot(backend_selector<backend::BACKEND> selector, std::int64_t n,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy, double c,
                       double s);

static inline void rot(backend_selector<backend::BACKEND> selector, std::int64_t n,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy, float c, float s);

static inline void rot(backend_selector<backend::BACKEND> selector, std::int64_t n,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &y, std::int64_t incy, double c, double s);

static inline void axpy(backend_selector<backend::BACKEND> selector, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);

static inline void axpy(backend_selector<backend::BACKEND> selector, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);

static inline void axpy(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy);

static inline void axpy(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy);

static inline void gerc(backend_selector<backend::BACKEND> selector, std::int64_t m, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda);

static inline void gerc(backend_selector<backend::BACKEND> selector, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda);

static inline void syr2k(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                         transpose trans, std::int64_t n, std::int64_t k, float alpha,
                         cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                         cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

static inline void syr2k(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                         transpose trans, std::int64_t n, std::int64_t k, double alpha,
                         cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                         cl::sycl::buffer<double, 1> &c, std::int64_t ldc);

static inline void syr2k(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                         transpose trans, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                         std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                         std::int64_t ldc);

static inline void syr2k(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                         transpose trans, std::int64_t n, std::int64_t k,
                         std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                         std::int64_t ldb, std::complex<double> beta,
                         cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

static inline void gemv(backend_selector<backend::BACKEND> selector, transpose trans,
                        std::int64_t m, std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy);

static inline void gemv(backend_selector<backend::BACKEND> selector, transpose trans,
                        std::int64_t m, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);

static inline void gemv(backend_selector<backend::BACKEND> selector, transpose trans,
                        std::int64_t m, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy);

static inline void gemv(backend_selector<backend::BACKEND> selector, transpose trans,
                        std::int64_t m, std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy);

static inline void her(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                       std::int64_t n, float alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &a,
                       std::int64_t lda);

static inline void her(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                       std::int64_t n, double alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &a,
                       std::int64_t lda);

static inline void hpr(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                       std::int64_t n, float alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &a);

static inline void hpr(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                       std::int64_t n, double alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &a);

static inline void iamin(backend_selector<backend::BACKEND> selector, std::int64_t n,
                         cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);

static inline void iamin(backend_selector<backend::BACKEND> selector, std::int64_t n,
                         cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);

static inline void iamin(backend_selector<backend::BACKEND> selector, std::int64_t n,
                         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);

static inline void iamin(backend_selector<backend::BACKEND> selector, std::int64_t n,
                         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);

static inline void gemm_batch(backend_selector<backend::BACKEND> selector, transpose transa,
                              transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                              float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<float, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, float beta,
                              cl::sycl::buffer<float, 1> &c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size);

static inline void gemm_batch(backend_selector<backend::BACKEND> selector, transpose transa,
                              transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                              double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<double, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, double beta,
                              cl::sycl::buffer<double, 1> &c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size);

static inline void gemm_batch(backend_selector<backend::BACKEND> selector, transpose transa,
                              transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                              std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                              cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size);

static inline void gemm_batch(backend_selector<backend::BACKEND> selector, transpose transa,
                              transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                              std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                              cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size);

static inline void spmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);

static inline void spmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);

static inline void gemm_bias(backend_selector<backend::BACKEND> selector, transpose transa,
                             transpose transb, offset offsetc, std::int64_t m, std::int64_t n,
                             std::int64_t k, float alpha, cl::sycl::buffer<int8_t, 1> &a,
                             std::int64_t lda, int8_t ao, cl::sycl::buffer<uint8_t, 1> &b,
                             std::int64_t ldb, uint8_t bo, float beta,
                             cl::sycl::buffer<int32_t, 1> &c, std::int64_t ldc,
                             cl::sycl::buffer<int32_t, 1> &co);

static inline void swap(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);

static inline void swap(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);

static inline void swap(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);

static inline void swap(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);

static inline void geru(backend_selector<backend::BACKEND> selector, std::int64_t m, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda);

static inline void geru(backend_selector<backend::BACKEND> selector, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda);

static inline void nrm2(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &result);

static inline void nrm2(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &result);

static inline void nrm2(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &result);

static inline void nrm2(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &result);

static inline void gemm(backend_selector<backend::BACKEND> selector, transpose transa,
                        transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                        float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                        cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

static inline void gemm(backend_selector<backend::BACKEND> selector, transpose transa,
                        transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                        double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                        cl::sycl::buffer<double, 1> &c, std::int64_t ldc);

static inline void gemm(backend_selector<backend::BACKEND> selector, transpose transa,
                        transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b,
                        std::int64_t ldb, std::complex<float> beta,
                        cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

static inline void gemm(backend_selector<backend::BACKEND> selector, transpose transa,
                        transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                        std::int64_t ldb, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

static inline void gemm(backend_selector<backend::BACKEND> selector, transpose transa,
                        transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                        cl::sycl::half alpha, cl::sycl::buffer<cl::sycl::half, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<cl::sycl::half, 1> &b, std::int64_t ldb, cl::sycl::half beta,
                        cl::sycl::buffer<cl::sycl::half, 1> &c, std::int64_t ldc);

static inline void gemm(backend_selector<backend::BACKEND> selector, transpose transa,
                        transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                        float alpha, cl::sycl::buffer<cl::sycl::half, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<cl::sycl::half, 1> &b, std::int64_t ldb, float beta,
                        cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

static inline void herk(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, std::int64_t n, std::int64_t k, float alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda, float beta,
                        cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

static inline void herk(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, std::int64_t n, std::int64_t k, double alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, double beta,
                        cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

static inline void ger(backend_selector<backend::BACKEND> selector, std::int64_t m, std::int64_t n,
                       float alpha, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda);

static inline void ger(backend_selector<backend::BACKEND> selector, std::int64_t m, std::int64_t n,
                       double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda);

static inline void trsm(backend_selector<backend::BACKEND> selector, side left_right,
                        uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                        std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb);

static inline void trsm(backend_selector<backend::BACKEND> selector, side left_right,
                        uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                        std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb);

static inline void trsm(backend_selector<backend::BACKEND> selector, side left_right,
                        uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);

static inline void trsm(backend_selector<backend::BACKEND> selector, side left_right,
                        uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);

static inline void dotu(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<float>, 1> &result);

static inline void dotu(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<double>, 1> &result);

static inline void hemm(backend_selector<backend::BACKEND> selector, side left_right,
                        uplo upper_lower, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc);

static inline void hemm(backend_selector<backend::BACKEND> selector, side left_right,
                        uplo upper_lower, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                        std::int64_t ldb, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

static inline void hpr2(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<float>, 1> &a);

static inline void hpr2(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<double>, 1> &a);

static inline void gbmv(backend_selector<backend::BACKEND> selector, transpose trans,
                        std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                        float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);

static inline void gbmv(backend_selector<backend::BACKEND> selector, transpose trans,
                        std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                        double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);

static inline void gbmv(backend_selector<backend::BACKEND> selector, transpose trans,
                        std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, std::complex<float> beta,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);

static inline void gbmv(backend_selector<backend::BACKEND> selector, transpose trans,
                        std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);

static inline void tbmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx);

static inline void tbmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx);

static inline void tbmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);

static inline void tbmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

static inline void symm(backend_selector<backend::BACKEND> selector, side left_right,
                        uplo upper_lower, std::int64_t m, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                        cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

static inline void symm(backend_selector<backend::BACKEND> selector, side left_right,
                        uplo upper_lower, std::int64_t m, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                        cl::sycl::buffer<double, 1> &c, std::int64_t ldc);

static inline void symm(backend_selector<backend::BACKEND> selector, side left_right,
                        uplo upper_lower, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc);

static inline void symm(backend_selector<backend::BACKEND> selector, side left_right,
                        uplo upper_lower, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                        std::int64_t ldb, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

static inline void dotc(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<float>, 1> &result);

static inline void dotc(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<double>, 1> &result);

static inline void syr(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                       std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &a, std::int64_t lda);

static inline void syr(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                       std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<double, 1> &a, std::int64_t lda);

static inline void trmm(backend_selector<backend::BACKEND> selector, side left_right,
                        uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                        std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb);

static inline void trmm(backend_selector<backend::BACKEND> selector, side left_right,
                        uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                        std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb);

static inline void trmm(backend_selector<backend::BACKEND> selector, side left_right,
                        uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);

static inline void trmm(backend_selector<backend::BACKEND> selector, side left_right,
                        uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);

static inline void rotmg(backend_selector<backend::BACKEND> selector,
                         cl::sycl::buffer<float, 1> &d1, cl::sycl::buffer<float, 1> &d2,
                         cl::sycl::buffer<float, 1> &x1, float y1,
                         cl::sycl::buffer<float, 1> &param);

static inline void rotmg(backend_selector<backend::BACKEND> selector,
                         cl::sycl::buffer<double, 1> &d1, cl::sycl::buffer<double, 1> &d2,
                         cl::sycl::buffer<double, 1> &x1, double y1,
                         cl::sycl::buffer<double, 1> &param);

static inline void tpsv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx);

static inline void tpsv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx);

static inline void tpsv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);

static inline void tpsv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

static inline void trsv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx);

static inline void trsv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx);

static inline void trsv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);

static inline void trsv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

static inline void copy(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);

static inline void copy(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);

static inline void copy(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);

static inline void copy(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);

static inline void hemv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy);

static inline void hemv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy);

static inline void gemmt(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                         transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                         float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                         cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

static inline void gemmt(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                         transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                         double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                         cl::sycl::buffer<double, 1> &c, std::int64_t ldc);

static inline void gemmt(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                         transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                         std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b,
                         std::int64_t ldb, std::complex<float> beta,
                         cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

static inline void gemmt(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                         transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                         std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                         std::int64_t ldb, std::complex<double> beta,
                         cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

static inline void sbmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy);

static inline void sbmv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, std::int64_t k, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);

static inline void asum(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &result);

static inline void asum(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &result);

static inline void asum(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &result);

static inline void asum(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &result);

static inline void tbsv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx);

static inline void tbsv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx);

static inline void tbsv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);

static inline void tbsv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

static inline void spr2(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<float, 1> &a);

static inline void spr2(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<double, 1> &a);

static inline void iamax(backend_selector<backend::BACKEND> selector, std::int64_t n,
                         cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);

static inline void iamax(backend_selector<backend::BACKEND> selector, std::int64_t n,
                         cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);

static inline void iamax(backend_selector<backend::BACKEND> selector, std::int64_t n,
                         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);

static inline void iamax(backend_selector<backend::BACKEND> selector, std::int64_t n,
                         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);

static inline void trsm_batch(backend_selector<backend::BACKEND> selector, side left_right,
                              uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                              std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                              std::int64_t lda, std::int64_t stride_a,
                              cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                              std::int64_t stride_b, std::int64_t batch_size);

static inline void trsm_batch(backend_selector<backend::BACKEND> selector, side left_right,
                              uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                              std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                              std::int64_t lda, std::int64_t stride_a,
                              cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                              std::int64_t stride_b, std::int64_t batch_size);

static inline void trsm_batch(backend_selector<backend::BACKEND> selector, side left_right,
                              uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                              std::int64_t n, std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);

static inline void trsm_batch(backend_selector<backend::BACKEND> selector, side left_right,
                              uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                              std::int64_t n, std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);

static inline void rotm(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<float, 1> &param);

static inline void rotm(backend_selector<backend::BACKEND> selector, std::int64_t n,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<double, 1> &param);

static inline void rotg(backend_selector<backend::BACKEND> selector, cl::sycl::buffer<float, 1> &a,
                        cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<float, 1> &c,
                        cl::sycl::buffer<float, 1> &s);

static inline void rotg(backend_selector<backend::BACKEND> selector, cl::sycl::buffer<double, 1> &a,
                        cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<double, 1> &c,
                        cl::sycl::buffer<double, 1> &s);

static inline void rotg(backend_selector<backend::BACKEND> selector,
                        cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<float, 1> &c,
                        cl::sycl::buffer<std::complex<float>, 1> &s);

static inline void rotg(backend_selector<backend::BACKEND> selector,
                        cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &b,
                        cl::sycl::buffer<double, 1> &c,
                        cl::sycl::buffer<std::complex<double>, 1> &s);

static inline void sdsdot(backend_selector<backend::BACKEND> selector, std::int64_t n, float sb,
                          cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                          cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                          cl::sycl::buffer<float, 1> &result);

static inline void her2k(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                         transpose trans, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, float beta,
                         cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

static inline void her2k(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                         transpose trans, std::int64_t n, std::int64_t k,
                         std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                         std::int64_t ldb, double beta,
                         cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

static inline void dot(backend_selector<backend::BACKEND> selector, std::int64_t n,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<float, 1> &result);

static inline void dot(backend_selector<backend::BACKEND> selector, std::int64_t n,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &result);

static inline void dot(backend_selector<backend::BACKEND> selector, std::int64_t n,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &result);

static inline void symv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy);

static inline void symv(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                        std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy);

// USM APIs

static inline cl::sycl::event syr2(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n, float alpha,
    const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
    std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event syr2(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n, double alpha,
    const double *x, std::int64_t incx, const double *y, std::int64_t incy, double *a,
    std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event scal(
    backend_selector<backend::BACKEND> selector, std::int64_t n, float alpha, float *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event scal(
    backend_selector<backend::BACKEND> selector, std::int64_t n, double alpha, double *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event scal(
    backend_selector<backend::BACKEND> selector, std::int64_t n, std::complex<float> alpha,
    std::complex<float> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event scal(
    backend_selector<backend::BACKEND> selector, std::int64_t n, std::complex<double> alpha,
    std::complex<double> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event scal(
    backend_selector<backend::BACKEND> selector, std::int64_t n, float alpha,
    std::complex<float> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event scal(
    backend_selector<backend::BACKEND> selector, std::int64_t n, double alpha,
    std::complex<double> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event trmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, const float *a, std::int64_t lda, float *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event trmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, const double *a, std::int64_t lda, double *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event trmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event trmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event tpmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, const float *a, float *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event tpmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, const double *a, double *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event tpmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, const std::complex<float> *a, std::complex<float> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event tpmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, const std::complex<double> *a, std::complex<double> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event spr(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                                  std::int64_t n, float alpha, const float *x, std::int64_t incx,
                                  float *a,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event spr(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                                  std::int64_t n, double alpha, const double *x, std::int64_t incx,
                                  double *a,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event hpmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *a, const std::complex<float> *x,
    std::int64_t incx, std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event hpmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n,
    std::complex<double> alpha, const std::complex<double> *a, const std::complex<double> *x,
    std::int64_t incx, std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event syrk(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, std::int64_t n,
    std::int64_t k, float alpha, const float *a, std::int64_t lda, float beta, float *c,
    std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event syrk(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, std::int64_t n,
    std::int64_t k, double alpha, const double *a, std::int64_t lda, double beta, double *c,
    std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event syrk(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, std::int64_t n,
    std::int64_t k, std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event syrk(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, std::int64_t n,
    std::int64_t k, std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event her2(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
    const std::complex<float> *y, std::int64_t incy, std::complex<float> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event her2(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n,
    std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
    const std::complex<double> *y, std::int64_t incy, std::complex<double> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event hbmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n, std::int64_t k,
    std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
    std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event hbmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n, std::int64_t k,
    std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
    std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event rot(backend_selector<backend::BACKEND> selector, std::int64_t n,
                                  std::complex<float> *x, std::int64_t incx, std::complex<float> *y,
                                  std::int64_t incy, float c, float s,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event rot(backend_selector<backend::BACKEND> selector, std::int64_t n,
                                  std::complex<double> *x, std::int64_t incx,
                                  std::complex<double> *y, std::int64_t incy, double c, double s,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event rot(backend_selector<backend::BACKEND> selector, std::int64_t n,
                                  float *x, std::int64_t incx, float *y, std::int64_t incy, float c,
                                  float s,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event rot(backend_selector<backend::BACKEND> selector, std::int64_t n,
                                  double *x, std::int64_t incx, double *y, std::int64_t incy,
                                  double c, double s,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event axpy(
    backend_selector<backend::BACKEND> selector, std::int64_t n, float alpha, const float *x,
    std::int64_t incx, float *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event axpy(
    backend_selector<backend::BACKEND> selector, std::int64_t n, double alpha, const double *x,
    std::int64_t incx, double *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event axpy(
    backend_selector<backend::BACKEND> selector, std::int64_t n, std::complex<float> alpha,
    const std::complex<float> *x, std::int64_t incx, std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event axpy(
    backend_selector<backend::BACKEND> selector, std::int64_t n, std::complex<double> alpha,
    const std::complex<double> *x, std::int64_t incx, std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event axpy_batch(
    backend_selector<backend::BACKEND> selector, std::int64_t *n, float *alpha, const float **x,
    std::int64_t *incx, float **y, std::int64_t *incy, std::int64_t group_count,
    std::int64_t *group_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event axpy_batch(
    backend_selector<backend::BACKEND> selector, std::int64_t *n, double *alpha, const double **x,
    std::int64_t *incx, double **y, std::int64_t *incy, std::int64_t group_count,
    std::int64_t *group_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event axpy_batch(
    backend_selector<backend::BACKEND> selector, std::int64_t *n, std::complex<float> *alpha,
    const std::complex<float> **x, std::int64_t *incx, std::complex<float> **y, std::int64_t *incy,
    std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event axpy_batch(
    backend_selector<backend::BACKEND> selector, std::int64_t *n, std::complex<double> *alpha,
    const std::complex<double> **x, std::int64_t *incx, std::complex<double> **y,
    std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gerc(
    backend_selector<backend::BACKEND> selector, std::int64_t m, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
    const std::complex<float> *y, std::int64_t incy, std::complex<float> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gerc(
    backend_selector<backend::BACKEND> selector, std::int64_t m, std::int64_t n,
    std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
    const std::complex<double> *y, std::int64_t incy, std::complex<double> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event syr2k(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, std::int64_t n,
    std::int64_t k, float alpha, const float *a, std::int64_t lda, const float *b, std::int64_t ldb,
    float beta, float *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event syr2k(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, std::int64_t n,
    std::int64_t k, double alpha, const double *a, std::int64_t lda, const double *b,
    std::int64_t ldb, double beta, double *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event syr2k(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, std::int64_t n,
    std::int64_t k, std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
    std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event syr2k(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, std::int64_t n,
    std::int64_t k, std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
    std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemv(
    backend_selector<backend::BACKEND> selector, transpose trans, std::int64_t m, std::int64_t n,
    float alpha, const float *a, std::int64_t lda, const float *x, std::int64_t incx, float beta,
    float *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemv(
    backend_selector<backend::BACKEND> selector, transpose trans, std::int64_t m, std::int64_t n,
    double alpha, const double *a, std::int64_t lda, const double *x, std::int64_t incx,
    double beta, double *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemv(
    backend_selector<backend::BACKEND> selector, transpose trans, std::int64_t m, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
    std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemv(
    backend_selector<backend::BACKEND> selector, transpose trans, std::int64_t m, std::int64_t n,
    std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
    std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event her(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                                  std::int64_t n, float alpha, const std::complex<float> *x,
                                  std::int64_t incx, std::complex<float> *a, std::int64_t lda,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event her(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                                  std::int64_t n, double alpha, const std::complex<double> *x,
                                  std::int64_t incx, std::complex<double> *a, std::int64_t lda,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event hpr(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                                  std::int64_t n, float alpha, const std::complex<float> *x,
                                  std::int64_t incx, std::complex<float> *a,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event hpr(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                                  std::int64_t n, double alpha, const std::complex<double> *x,
                                  std::int64_t incx, std::complex<double> *a,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event iamin(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const float *x, std::int64_t incx,
    std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event iamin(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const double *x, std::int64_t incx,
    std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event iamin(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const std::complex<float> *x,
    std::int64_t incx, std::int64_t *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event iamin(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const std::complex<double> *x,
    std::int64_t incx, std::int64_t *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemm_batch(
    backend_selector<backend::BACKEND> selector, transpose *transa, transpose *transb,
    std::int64_t *m, std::int64_t *n, std::int64_t *k, float *alpha, const float **a,
    std::int64_t *lda, const float **b, std::int64_t *ldb, float *beta, float **c,
    std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemm_batch(
    backend_selector<backend::BACKEND> selector, transpose *transa, transpose *transb,
    std::int64_t *m, std::int64_t *n, std::int64_t *k, double *alpha, const double **a,
    std::int64_t *lda, const double **b, std::int64_t *ldb, double *beta, double **c,
    std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemm_batch(
    backend_selector<backend::BACKEND> selector, transpose *transa, transpose *transb,
    std::int64_t *m, std::int64_t *n, std::int64_t *k, std::complex<float> *alpha,
    const std::complex<float> **a, std::int64_t *lda, const std::complex<float> **b,
    std::int64_t *ldb, std::complex<float> *beta, std::complex<float> **c, std::int64_t *ldc,
    std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemm_batch(
    backend_selector<backend::BACKEND> selector, transpose *transa, transpose *transb,
    std::int64_t *m, std::int64_t *n, std::int64_t *k, std::complex<double> *alpha,
    const std::complex<double> **a, std::int64_t *lda, const std::complex<double> **b,
    std::int64_t *ldb, std::complex<double> *beta, std::complex<double> **c, std::int64_t *ldc,
    std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemm_batch(
    backend_selector<backend::BACKEND> selector, transpose transa, transpose transb, std::int64_t m,
    std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
    std::int64_t stride_a, const float *b, std::int64_t ldb, std::int64_t stride_b, float beta,
    float *c, std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemm_batch(
    backend_selector<backend::BACKEND> selector, transpose transa, transpose transb, std::int64_t m,
    std::int64_t n, std::int64_t k, double alpha, const double *a, std::int64_t lda,
    std::int64_t stride_a, const double *b, std::int64_t ldb, std::int64_t stride_b, double beta,
    double *c, std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemm_batch(
    backend_selector<backend::BACKEND> selector, transpose transa, transpose transb, std::int64_t m,
    std::int64_t n, std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
    std::int64_t lda, std::int64_t stride_a, const std::complex<float> *b, std::int64_t ldb,
    std::int64_t stride_b, std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
    std::int64_t stride_c, std::int64_t batch_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemm_batch(
    backend_selector<backend::BACKEND> selector, transpose transa, transpose transb, std::int64_t m,
    std::int64_t n, std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
    std::int64_t lda, std::int64_t stride_a, const std::complex<double> *b, std::int64_t ldb,
    std::int64_t stride_b, std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
    std::int64_t stride_c, std::int64_t batch_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event spmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n, float alpha,
    const float *a, const float *x, std::int64_t incx, float beta, float *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event spmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n, double alpha,
    const double *a, const double *x, std::int64_t incx, double beta, double *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event swap(
    backend_selector<backend::BACKEND> selector, std::int64_t n, float *x, std::int64_t incx,
    float *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event swap(
    backend_selector<backend::BACKEND> selector, std::int64_t n, double *x, std::int64_t incx,
    double *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event swap(
    backend_selector<backend::BACKEND> selector, std::int64_t n, std::complex<float> *x,
    std::int64_t incx, std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event swap(
    backend_selector<backend::BACKEND> selector, std::int64_t n, std::complex<double> *x,
    std::int64_t incx, std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event geru(
    backend_selector<backend::BACKEND> selector, std::int64_t m, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
    const std::complex<float> *y, std::int64_t incy, std::complex<float> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event geru(
    backend_selector<backend::BACKEND> selector, std::int64_t m, std::int64_t n,
    std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
    const std::complex<double> *y, std::int64_t incy, std::complex<double> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event nrm2(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const std::complex<float> *x,
    std::int64_t incx, float *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event nrm2(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const std::complex<double> *x,
    std::int64_t incx, double *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event nrm2(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const float *x, std::int64_t incx,
    float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event nrm2(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const double *x, std::int64_t incx,
    double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemm(
    backend_selector<backend::BACKEND> selector, transpose transa, transpose transb, std::int64_t m,
    std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda, const float *b,
    std::int64_t ldb, float beta, float *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemm(
    backend_selector<backend::BACKEND> selector, transpose transa, transpose transb, std::int64_t m,
    std::int64_t n, std::int64_t k, double alpha, const double *a, std::int64_t lda,
    const double *b, std::int64_t ldb, double beta, double *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemm(
    backend_selector<backend::BACKEND> selector, transpose transa, transpose transb, std::int64_t m,
    std::int64_t n, std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
    std::int64_t lda, const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
    std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemm(
    backend_selector<backend::BACKEND> selector, transpose transa, transpose transb, std::int64_t m,
    std::int64_t n, std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
    std::int64_t lda, const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
    std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event herk(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, std::int64_t n,
    std::int64_t k, float alpha, const std::complex<float> *a, std::int64_t lda, float beta,
    std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event herk(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, std::int64_t n,
    std::int64_t k, double alpha, const std::complex<double> *a, std::int64_t lda, double beta,
    std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event ger(backend_selector<backend::BACKEND> selector, std::int64_t m,
                                  std::int64_t n, float alpha, const float *x, std::int64_t incx,
                                  const float *y, std::int64_t incy, float *a, std::int64_t lda,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event ger(backend_selector<backend::BACKEND> selector, std::int64_t m,
                                  std::int64_t n, double alpha, const double *x, std::int64_t incx,
                                  const double *y, std::int64_t incy, double *a, std::int64_t lda,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event trsm(
    backend_selector<backend::BACKEND> selector, side left_right, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t m, std::int64_t n, float alpha, const float *a, std::int64_t lda,
    float *b, std::int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event trsm(
    backend_selector<backend::BACKEND> selector, side left_right, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t m, std::int64_t n, double alpha, const double *a, std::int64_t lda,
    double *b, std::int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event trsm(
    backend_selector<backend::BACKEND> selector, side left_right, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
    const std::complex<float> *a, std::int64_t lda, std::complex<float> *b, std::int64_t ldb,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event trsm(
    backend_selector<backend::BACKEND> selector, side left_right, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
    const std::complex<double> *a, std::int64_t lda, std::complex<double> *b, std::int64_t ldb,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event dotu(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const std::complex<float> *x,
    std::int64_t incx, const std::complex<float> *y, std::int64_t incy, std::complex<float> *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event dotu(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const std::complex<double> *x,
    std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
    std::complex<double> *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event hemm(
    backend_selector<backend::BACKEND> selector, side left_right, uplo upper_lower, std::int64_t m,
    std::int64_t n, std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
    std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event hemm(
    backend_selector<backend::BACKEND> selector, side left_right, uplo upper_lower, std::int64_t m,
    std::int64_t n, std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
    std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event hpr2(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
    const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event hpr2(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n,
    std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
    const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gbmv(
    backend_selector<backend::BACKEND> selector, transpose trans, std::int64_t m, std::int64_t n,
    std::int64_t kl, std::int64_t ku, float alpha, const float *a, std::int64_t lda, const float *x,
    std::int64_t incx, float beta, float *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gbmv(
    backend_selector<backend::BACKEND> selector, transpose trans, std::int64_t m, std::int64_t n,
    std::int64_t kl, std::int64_t ku, double alpha, const double *a, std::int64_t lda,
    const double *x, std::int64_t incx, double beta, double *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gbmv(
    backend_selector<backend::BACKEND> selector, transpose trans, std::int64_t m, std::int64_t n,
    std::int64_t kl, std::int64_t ku, std::complex<float> alpha, const std::complex<float> *a,
    std::int64_t lda, const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
    std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gbmv(
    backend_selector<backend::BACKEND> selector, transpose trans, std::int64_t m, std::int64_t n,
    std::int64_t kl, std::int64_t ku, std::complex<double> alpha, const std::complex<double> *a,
    std::int64_t lda, const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
    std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event tbmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, std::int64_t k, const float *a, std::int64_t lda, float *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event tbmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, std::int64_t k, const double *a, std::int64_t lda, double *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event tbmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, std::int64_t k, const std::complex<float> *a, std::int64_t lda,
    std::complex<float> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event tbmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, std::int64_t k, const std::complex<double> *a, std::int64_t lda,
    std::complex<double> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event symm(
    backend_selector<backend::BACKEND> selector, side left_right, uplo upper_lower, std::int64_t m,
    std::int64_t n, float alpha, const float *a, std::int64_t lda, const float *b, std::int64_t ldb,
    float beta, float *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event symm(
    backend_selector<backend::BACKEND> selector, side left_right, uplo upper_lower, std::int64_t m,
    std::int64_t n, double alpha, const double *a, std::int64_t lda, const double *b,
    std::int64_t ldb, double beta, double *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event symm(
    backend_selector<backend::BACKEND> selector, side left_right, uplo upper_lower, std::int64_t m,
    std::int64_t n, std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
    std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event symm(
    backend_selector<backend::BACKEND> selector, side left_right, uplo upper_lower, std::int64_t m,
    std::int64_t n, std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
    std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event dotc(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const std::complex<float> *x,
    std::int64_t incx, const std::complex<float> *y, std::int64_t incy, std::complex<float> *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event dotc(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const std::complex<double> *x,
    std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
    std::complex<double> *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event syr(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                                  std::int64_t n, float alpha, const float *x, std::int64_t incx,
                                  float *a, std::int64_t lda,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event syr(backend_selector<backend::BACKEND> selector, uplo upper_lower,
                                  std::int64_t n, double alpha, const double *x, std::int64_t incx,
                                  double *a, std::int64_t lda,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event trmm(
    backend_selector<backend::BACKEND> selector, side left_right, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t m, std::int64_t n, float alpha, const float *a, std::int64_t lda,
    float *b, std::int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event trmm(
    backend_selector<backend::BACKEND> selector, side left_right, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t m, std::int64_t n, double alpha, const double *a, std::int64_t lda,
    double *b, std::int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event trmm(
    backend_selector<backend::BACKEND> selector, side left_right, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
    const std::complex<float> *a, std::int64_t lda, std::complex<float> *b, std::int64_t ldb,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event trmm(
    backend_selector<backend::BACKEND> selector, side left_right, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
    const std::complex<double> *a, std::int64_t lda, std::complex<double> *b, std::int64_t ldb,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event rotmg(
    backend_selector<backend::BACKEND> selector, float *d1, float *d2, float *x1, float y1,
    float *param, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event rotmg(
    backend_selector<backend::BACKEND> selector, double *d1, double *d2, double *x1, double y1,
    double *param, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event tpsv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, const float *a, float *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event tpsv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, const double *a, double *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event tpsv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, const std::complex<float> *a, std::complex<float> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event tpsv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, const std::complex<double> *a, std::complex<double> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event trsv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, const float *a, std::int64_t lda, float *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event trsv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, const double *a, std::int64_t lda, double *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event trsv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event trsv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event copy(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const float *x, std::int64_t incx,
    float *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event copy(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const double *x, std::int64_t incx,
    double *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event copy(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const std::complex<float> *x,
    std::int64_t incx, std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event copy(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const std::complex<double> *x,
    std::int64_t incx, std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event hemv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
    std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event hemv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n,
    std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
    std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemmt(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose transa,
    transpose transb, std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
    const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemmt(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose transa,
    transpose transb, std::int64_t n, std::int64_t k, double alpha, const double *a,
    std::int64_t lda, const double *b, std::int64_t ldb, double beta, double *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemmt(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose transa,
    transpose transb, std::int64_t n, std::int64_t k, std::complex<float> alpha,
    const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b, std::int64_t ldb,
    std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event gemmt(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose transa,
    transpose transb, std::int64_t n, std::int64_t k, std::complex<double> alpha,
    const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
    std::int64_t ldb, std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event sbmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n, std::int64_t k,
    float alpha, const float *a, std::int64_t lda, const float *x, std::int64_t incx, float beta,
    float *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event sbmv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n, std::int64_t k,
    double alpha, const double *a, std::int64_t lda, const double *x, std::int64_t incx,
    double beta, double *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event asum(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const std::complex<float> *x,
    std::int64_t incx, float *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event asum(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const std::complex<double> *x,
    std::int64_t incx, double *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event asum(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const float *x, std::int64_t incx,
    float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event asum(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const double *x, std::int64_t incx,
    double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event tbsv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, std::int64_t k, const float *a, std::int64_t lda, float *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event tbsv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, std::int64_t k, const double *a, std::int64_t lda, double *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event tbsv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, std::int64_t k, const std::complex<float> *a, std::int64_t lda,
    std::complex<float> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event tbsv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t n, std::int64_t k, const std::complex<double> *a, std::int64_t lda,
    std::complex<double> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event spr2(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n, float alpha,
    const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event spr2(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n, double alpha,
    const double *x, std::int64_t incx, const double *y, std::int64_t incy, double *a,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event iamax(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const float *x, std::int64_t incx,
    std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event iamax(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const double *x, std::int64_t incx,
    std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event iamax(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const std::complex<float> *x,
    std::int64_t incx, std::int64_t *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event iamax(
    backend_selector<backend::BACKEND> selector, std::int64_t n, const std::complex<double> *x,
    std::int64_t incx, std::int64_t *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event rotm(
    backend_selector<backend::BACKEND> selector, std::int64_t n, float *x, std::int64_t incx,
    float *y, std::int64_t incy, float *param,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event rotm(
    backend_selector<backend::BACKEND> selector, std::int64_t n, double *x, std::int64_t incx,
    double *y, std::int64_t incy, double *param,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event rotg(
    backend_selector<backend::BACKEND> selector, float *a, float *b, float *c, float *s,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event rotg(
    backend_selector<backend::BACKEND> selector, double *a, double *b, double *c, double *s,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event rotg(
    backend_selector<backend::BACKEND> selector, std::complex<float> *a, std::complex<float> *b,
    float *c, std::complex<float> *s,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event rotg(
    backend_selector<backend::BACKEND> selector, std::complex<double> *a, std::complex<double> *b,
    double *c, std::complex<double> *s,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event sdsdot(
    backend_selector<backend::BACKEND> selector, std::int64_t n, float sb, const float *x,
    std::int64_t incx, const float *y, std::int64_t incy, float *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event her2k(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, std::int64_t n,
    std::int64_t k, std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *b, std::int64_t ldb, float beta, std::complex<float> *c,
    std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event her2k(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, transpose trans, std::int64_t n,
    std::int64_t k, std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *b, std::int64_t ldb, double beta, std::complex<double> *c,
    std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event dot(backend_selector<backend::BACKEND> selector, std::int64_t n,
                                  const float *x, std::int64_t incx, const float *y,
                                  std::int64_t incy, float *result,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event dot(backend_selector<backend::BACKEND> selector, std::int64_t n,
                                  const double *x, std::int64_t incx, const double *y,
                                  std::int64_t incy, double *result,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event dot(backend_selector<backend::BACKEND> selector, std::int64_t n,
                                  const float *x, std::int64_t incx, const float *y,
                                  std::int64_t incy, double *result,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event symv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n, float alpha,
    const float *a, std::int64_t lda, const float *x, std::int64_t incx, float beta, float *y,
    std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

static inline cl::sycl::event symv(
    backend_selector<backend::BACKEND> selector, uplo upper_lower, std::int64_t n, double alpha,
    const double *a, std::int64_t lda, const double *x, std::int64_t incx, double beta, double *y,
    std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
