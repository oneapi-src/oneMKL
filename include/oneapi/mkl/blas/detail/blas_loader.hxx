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

ONEMKL_EXPORT void herk(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, std::int64_t n, std::int64_t k, float alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda, float beta,
                        cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
ONEMKL_EXPORT void herk(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, std::int64_t n, std::int64_t k, double alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, double beta,
                        cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void scal(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        float alpha, cl::sycl::buffer<float, 1> &x, std::int64_t incx);
ONEMKL_EXPORT void scal(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx);
ONEMKL_EXPORT void scal(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx);
ONEMKL_EXPORT void scal(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx);
ONEMKL_EXPORT void scal(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        float alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx);
ONEMKL_EXPORT void scal(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        double alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx);

ONEMKL_EXPORT void trmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx);
ONEMKL_EXPORT void trmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx);
ONEMKL_EXPORT void trmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
ONEMKL_EXPORT void trmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void tpmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx);
ONEMKL_EXPORT void tpmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx);
ONEMKL_EXPORT void tpmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
ONEMKL_EXPORT void tpmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void spr(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                       std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &a);
ONEMKL_EXPORT void spr(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                       std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<double, 1> &a);

ONEMKL_EXPORT void gemm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                              transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                              float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<float, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, float beta,
                              cl::sycl::buffer<float, 1> &c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size);
ONEMKL_EXPORT void gemm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                              transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                              double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<double, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, double beta,
                              cl::sycl::buffer<double, 1> &c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size);
ONEMKL_EXPORT void gemm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                              transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                              std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                              cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size);
ONEMKL_EXPORT void gemm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                              transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                              std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                              cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size);

ONEMKL_EXPORT void syrk(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, std::int64_t n, std::int64_t k, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda, float beta,
                        cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
ONEMKL_EXPORT void syrk(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, std::int64_t n, std::int64_t k, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda, double beta,
                        cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
ONEMKL_EXPORT void syrk(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc);
ONEMKL_EXPORT void syrk(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, std::int64_t n, std::int64_t k, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc);

ONEMKL_EXPORT void her2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);
ONEMKL_EXPORT void her2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);

ONEMKL_EXPORT void hbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, std::int64_t k, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy);
ONEMKL_EXPORT void hbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, std::int64_t k, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy);

ONEMKL_EXPORT void rot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy, float c,
                       float s);
ONEMKL_EXPORT void rot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy, double c,
                       double s);
ONEMKL_EXPORT void rot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy, float c, float s);
ONEMKL_EXPORT void rot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &y, std::int64_t incy, double c, double s);

ONEMKL_EXPORT void axpy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        float alpha, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);
ONEMKL_EXPORT void axpy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);
ONEMKL_EXPORT void axpy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy);
ONEMKL_EXPORT void axpy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy);

ONEMKL_EXPORT void gerc(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);
ONEMKL_EXPORT void gerc(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);

ONEMKL_EXPORT void syr2k(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                         transpose trans, std::int64_t n, std::int64_t k, float alpha,
                         cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                         cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
ONEMKL_EXPORT void syr2k(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                         transpose trans, std::int64_t n, std::int64_t k, double alpha,
                         cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                         cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
ONEMKL_EXPORT void syr2k(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                         transpose trans, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                         std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                         std::int64_t ldc);
ONEMKL_EXPORT void syr2k(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                         transpose trans, std::int64_t n, std::int64_t k,
                         std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                         std::int64_t ldb, std::complex<double> beta,
                         cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void gemv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans,
                        std::int64_t m, std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy);
ONEMKL_EXPORT void gemv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans,
                        std::int64_t m, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);
ONEMKL_EXPORT void gemv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans,
                        std::int64_t m, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy);
ONEMKL_EXPORT void gemv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans,
                        std::int64_t m, std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy);

ONEMKL_EXPORT void her(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                       std::int64_t n, float alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &a,
                       std::int64_t lda);
ONEMKL_EXPORT void her(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                       std::int64_t n, double alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &a,
                       std::int64_t lda);

ONEMKL_EXPORT void hpr(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                       std::int64_t n, float alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &a);
ONEMKL_EXPORT void hpr(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                       std::int64_t n, double alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &a);

ONEMKL_EXPORT void gemm_bias(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                             transpose transb, offset offsetc, std::int64_t m, std::int64_t n,
                             std::int64_t k, float alpha, cl::sycl::buffer<int8_t, 1> &a,
                             std::int64_t lda, int8_t ao, cl::sycl::buffer<uint8_t, 1> &b,
                             std::int64_t ldb, uint8_t bo, float beta,
                             cl::sycl::buffer<int32_t, 1> &c, std::int64_t ldc,
                             cl::sycl::buffer<int32_t, 1> &co);

ONEMKL_EXPORT void iamin(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);
ONEMKL_EXPORT void iamin(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);
ONEMKL_EXPORT void iamin(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);
ONEMKL_EXPORT void iamin(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);

ONEMKL_EXPORT void hpmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy);
ONEMKL_EXPORT void hpmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy);

ONEMKL_EXPORT void spmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);
ONEMKL_EXPORT void spmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void rotmg(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                         cl::sycl::buffer<float, 1> &d1, cl::sycl::buffer<float, 1> &d2,
                         cl::sycl::buffer<float, 1> &x1, float y1,
                         cl::sycl::buffer<float, 1> &param);
ONEMKL_EXPORT void rotmg(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                         cl::sycl::buffer<double, 1> &d1, cl::sycl::buffer<double, 1> &d2,
                         cl::sycl::buffer<double, 1> &x1, double y1,
                         cl::sycl::buffer<double, 1> &param);

ONEMKL_EXPORT void swap(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);
ONEMKL_EXPORT void swap(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);
ONEMKL_EXPORT void swap(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
ONEMKL_EXPORT void swap(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void geru(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);
ONEMKL_EXPORT void geru(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);

ONEMKL_EXPORT void nrm2(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &result);
ONEMKL_EXPORT void nrm2(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &result);
ONEMKL_EXPORT void nrm2(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &result);
ONEMKL_EXPORT void nrm2(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &result);

ONEMKL_EXPORT void gemmt(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                         transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                         float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                         cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
ONEMKL_EXPORT void gemmt(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                         transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                         double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                         cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
ONEMKL_EXPORT void gemmt(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                         transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                         std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b,
                         std::int64_t ldb, std::complex<float> beta,
                         cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
ONEMKL_EXPORT void gemmt(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                         transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                         std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                         std::int64_t ldb, std::complex<double> beta,
                         cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void gemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                        transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                        float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                        cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
ONEMKL_EXPORT void gemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                        transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                        double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                        cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
ONEMKL_EXPORT void gemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                        transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b,
                        std::int64_t ldb, std::complex<float> beta,
                        cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
ONEMKL_EXPORT void gemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                        transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                        std::int64_t ldb, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
ONEMKL_EXPORT void gemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                        transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                        cl::sycl::half alpha, cl::sycl::buffer<cl::sycl::half, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<cl::sycl::half, 1> &b, std::int64_t ldb, cl::sycl::half beta,
                        cl::sycl::buffer<cl::sycl::half, 1> &c, std::int64_t ldc);
ONEMKL_EXPORT void gemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                        transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                        float alpha, cl::sycl::buffer<cl::sycl::half, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<cl::sycl::half, 1> &b, std::int64_t ldb, float beta,
                        cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void syr2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda);
ONEMKL_EXPORT void syr2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda);

ONEMKL_EXPORT void ger(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m,
                       std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda);
ONEMKL_EXPORT void ger(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m,
                       std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda);

ONEMKL_EXPORT void trsm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                        uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                        std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb);
ONEMKL_EXPORT void trsm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                        uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                        std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb);
ONEMKL_EXPORT void trsm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                        uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);
ONEMKL_EXPORT void trsm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                        uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);

ONEMKL_EXPORT void dotu(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<float>, 1> &result);
ONEMKL_EXPORT void dotu(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<double>, 1> &result);

ONEMKL_EXPORT void hemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                        uplo upper_lower, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc);
ONEMKL_EXPORT void hemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                        uplo upper_lower, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                        std::int64_t ldb, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void hpr2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<float>, 1> &a);
ONEMKL_EXPORT void hpr2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<double>, 1> &a);

ONEMKL_EXPORT void gbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans,
                        std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                        float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);
ONEMKL_EXPORT void gbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans,
                        std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                        double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);
ONEMKL_EXPORT void gbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans,
                        std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, std::complex<float> beta,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
ONEMKL_EXPORT void gbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans,
                        std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void tbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx);
ONEMKL_EXPORT void tbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx);
ONEMKL_EXPORT void tbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
ONEMKL_EXPORT void tbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void symm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                        uplo upper_lower, std::int64_t m, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                        cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
ONEMKL_EXPORT void symm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                        uplo upper_lower, std::int64_t m, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                        cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
ONEMKL_EXPORT void symm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                        uplo upper_lower, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc);
ONEMKL_EXPORT void symm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                        uplo upper_lower, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                        std::int64_t ldb, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void dotc(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<float>, 1> &result);
ONEMKL_EXPORT void dotc(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<double>, 1> &result);

ONEMKL_EXPORT void syr(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                       std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &a, std::int64_t lda);
ONEMKL_EXPORT void syr(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                       std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<double, 1> &a, std::int64_t lda);

ONEMKL_EXPORT void trmm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                        uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                        std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb);
ONEMKL_EXPORT void trmm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                        uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                        std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb);
ONEMKL_EXPORT void trmm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                        uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);
ONEMKL_EXPORT void trmm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                        uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);

ONEMKL_EXPORT void symv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy);
ONEMKL_EXPORT void symv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void tpsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx);
ONEMKL_EXPORT void tpsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx);
ONEMKL_EXPORT void tpsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
ONEMKL_EXPORT void tpsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void trsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx);
ONEMKL_EXPORT void trsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx);
ONEMKL_EXPORT void trsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
ONEMKL_EXPORT void trsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void copy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);
ONEMKL_EXPORT void copy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);
ONEMKL_EXPORT void copy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
ONEMKL_EXPORT void copy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void hemv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy);
ONEMKL_EXPORT void hemv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy);

ONEMKL_EXPORT void iamax(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);
ONEMKL_EXPORT void iamax(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);
ONEMKL_EXPORT void iamax(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);
ONEMKL_EXPORT void iamax(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);

ONEMKL_EXPORT void sbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy);
ONEMKL_EXPORT void sbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, std::int64_t k, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void asum(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &result);
ONEMKL_EXPORT void asum(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &result);
ONEMKL_EXPORT void asum(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &result);
ONEMKL_EXPORT void asum(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &result);

ONEMKL_EXPORT void tbsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx);
ONEMKL_EXPORT void tbsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx);
ONEMKL_EXPORT void tbsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
ONEMKL_EXPORT void tbsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void spr2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<float, 1> &a);
ONEMKL_EXPORT void spr2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                        std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<double, 1> &a);

ONEMKL_EXPORT void trsm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                              uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                              std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                              std::int64_t lda, std::int64_t stride_a,
                              cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                              std::int64_t stride_b, std::int64_t batch_size);
ONEMKL_EXPORT void trsm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                              uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                              std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                              std::int64_t lda, std::int64_t stride_a,
                              cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                              std::int64_t stride_b, std::int64_t batch_size);
ONEMKL_EXPORT void trsm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                              uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                              std::int64_t n, std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);
ONEMKL_EXPORT void trsm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                              uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                              std::int64_t n, std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);

ONEMKL_EXPORT void rotm(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<float, 1> &param);
ONEMKL_EXPORT void rotm(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<double, 1> &param);

ONEMKL_EXPORT void dot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<float, 1> &result);
ONEMKL_EXPORT void dot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &result);
ONEMKL_EXPORT void dot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &result);

ONEMKL_EXPORT void sdsdot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                          float sb, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                          cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                          cl::sycl::buffer<float, 1> &result);

ONEMKL_EXPORT void her2k(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                         transpose trans, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, float beta,
                         cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
ONEMKL_EXPORT void her2k(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                         transpose trans, std::int64_t n, std::int64_t k,
                         std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                         std::int64_t ldb, double beta,
                         cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void rotg(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                        cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &b,
                        cl::sycl::buffer<float, 1> &c, cl::sycl::buffer<float, 1> &s);
ONEMKL_EXPORT void rotg(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                        cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &b,
                        cl::sycl::buffer<double, 1> &c, cl::sycl::buffer<double, 1> &s);
ONEMKL_EXPORT void rotg(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                        cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<float, 1> &c,
                        cl::sycl::buffer<std::complex<float>, 1> &s);
ONEMKL_EXPORT void rotg(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                        cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &b,
                        cl::sycl::buffer<double, 1> &c,
                        cl::sycl::buffer<std::complex<double>, 1> &s);

// USM APIs

ONEMKL_EXPORT cl::sycl::event herk(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    std::int64_t n, std::int64_t k, float alpha, const std::complex<float> *a, std::int64_t lda,
    float beta, std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event herk(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    std::int64_t n, std::int64_t k, double alpha, const std::complex<double> *a, std::int64_t lda,
    double beta, std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event scal(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, float alpha, float *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event scal(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, double alpha, double *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event scal(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
    std::complex<float> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event scal(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
    std::complex<double> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event scal(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, float alpha,
    std::complex<float> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event scal(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, double alpha,
    std::complex<double> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event trmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, const float *a, std::int64_t lda, float *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event trmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, const double *a, std::int64_t lda, double *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event trmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, const std::complex<float> *a, std::int64_t lda,
    std::complex<float> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event trmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, const std::complex<double> *a, std::int64_t lda,
    std::complex<double> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event tpmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, const float *a, float *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event tpmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, const double *a, double *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event tpmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, const std::complex<float> *a, std::complex<float> *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event tpmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, const std::complex<double> *a, std::complex<double> *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event spr(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                                  uplo upper_lower, std::int64_t n, float alpha, const float *x,
                                  std::int64_t incx, float *a,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event spr(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                                  uplo upper_lower, std::int64_t n, double alpha, const double *x,
                                  std::int64_t incx, double *a,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event gemm_batch(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose *transa, transpose *transb,
    std::int64_t *m, std::int64_t *n, std::int64_t *k, float *alpha, const float **a,
    std::int64_t *lda, const float **b, std::int64_t *ldb, float *beta, float **c,
    std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gemm_batch(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose *transa, transpose *transb,
    std::int64_t *m, std::int64_t *n, std::int64_t *k, double *alpha, const double **a,
    std::int64_t *lda, const double **b, std::int64_t *ldb, double *beta, double **c,
    std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gemm_batch(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose *transa, transpose *transb,
    std::int64_t *m, std::int64_t *n, std::int64_t *k, std::complex<float> *alpha,
    const std::complex<float> **a, std::int64_t *lda, const std::complex<float> **b,
    std::int64_t *ldb, std::complex<float> *beta, std::complex<float> **c, std::int64_t *ldc,
    std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gemm_batch(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose *transa, transpose *transb,
    std::int64_t *m, std::int64_t *n, std::int64_t *k, std::complex<double> *alpha,
    const std::complex<double> **a, std::int64_t *lda, const std::complex<double> **b,
    std::int64_t *ldb, std::complex<double> *beta, std::complex<double> **c, std::int64_t *ldc,
    std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gemm_batch(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa, transpose transb,
    std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
    std::int64_t stride_a, const float *b, std::int64_t ldb, std::int64_t stride_b, float beta,
    float *c, std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gemm_batch(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa, transpose transb,
    std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double *a, std::int64_t lda,
    std::int64_t stride_a, const double *b, std::int64_t ldb, std::int64_t stride_b, double beta,
    double *c, std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gemm_batch(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa, transpose transb,
    std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
    const std::complex<float> *a, std::int64_t lda, std::int64_t stride_a,
    const std::complex<float> *b, std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
    std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gemm_batch(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa, transpose transb,
    std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
    const std::complex<double> *a, std::int64_t lda, std::int64_t stride_a,
    const std::complex<double> *b, std::int64_t ldb, std::int64_t stride_b,
    std::complex<double> beta, std::complex<double> *c, std::int64_t ldc, std::int64_t stride_c,
    std::int64_t batch_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event syrk(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda, float beta,
    float *c, std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event syrk(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    std::int64_t n, std::int64_t k, double alpha, const double *a, std::int64_t lda, double beta,
    double *c, std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event syrk(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    std::int64_t n, std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
    std::int64_t lda, std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event syrk(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    std::int64_t n, std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
    std::int64_t lda, std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event her2(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
    const std::complex<float> *y, std::int64_t incy, std::complex<float> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event her2(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
    const std::complex<double> *y, std::int64_t incy, std::complex<double> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event hbmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    std::int64_t k, std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
    std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event hbmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    std::int64_t k, std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
    std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event rot(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                                  std::int64_t n, std::complex<float> *x, std::int64_t incx,
                                  std::complex<float> *y, std::int64_t incy, float c, float s,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event rot(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                                  std::int64_t n, std::complex<double> *x, std::int64_t incx,
                                  std::complex<double> *y, std::int64_t incy, double c, double s,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event rot(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                                  std::int64_t n, float *x, std::int64_t incx, float *y,
                                  std::int64_t incy, float c, float s,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event rot(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                                  std::int64_t n, double *x, std::int64_t incx, double *y,
                                  std::int64_t incy, double c, double s,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event axpy(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, float alpha, const float *x,
    std::int64_t incx, float *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event axpy(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, double alpha,
    const double *x, std::int64_t incx, double *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event axpy(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
    const std::complex<float> *x, std::int64_t incx, std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event axpy(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
    const std::complex<double> *x, std::int64_t incx, std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event axpy_batch(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t *n, float *alpha,
    const float **x, std::int64_t *incx, float **y, std::int64_t *incy, std::int64_t group_count,
    std::int64_t *group_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event axpy_batch(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t *n, double *alpha,
    const double **x, std::int64_t *incx, double **y, std::int64_t *incy, std::int64_t group_count,
    std::int64_t *group_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event axpy_batch(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t *n, std::complex<float> *alpha,
    const std::complex<float> **x, std::int64_t *incx, std::complex<float> **y, std::int64_t *incy,
    std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event axpy_batch(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t *n,
    std::complex<double> *alpha, const std::complex<double> **x, std::int64_t *incx,
    std::complex<double> **y, std::int64_t *incy, std::int64_t group_count,
    std::int64_t *group_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event gerc(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
    const std::complex<float> *y, std::int64_t incy, std::complex<float> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gerc(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
    std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
    const std::complex<double> *y, std::int64_t incy, std::complex<double> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event syr2k(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda, const float *b,
    std::int64_t ldb, float beta, float *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event syr2k(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    std::int64_t n, std::int64_t k, double alpha, const double *a, std::int64_t lda,
    const double *b, std::int64_t ldb, double beta, double *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event syr2k(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    std::int64_t n, std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
    std::int64_t lda, const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
    std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event syr2k(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    std::int64_t n, std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
    std::int64_t lda, const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
    std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event gemv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans, std::int64_t m,
    std::int64_t n, float alpha, const float *a, std::int64_t lda, const float *x,
    std::int64_t incx, float beta, float *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gemv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans, std::int64_t m,
    std::int64_t n, double alpha, const double *a, std::int64_t lda, const double *x,
    std::int64_t incx, double beta, double *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gemv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans, std::int64_t m,
    std::int64_t n, std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
    std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gemv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans, std::int64_t m,
    std::int64_t n, std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
    std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event her(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                                  uplo upper_lower, std::int64_t n, float alpha,
                                  const std::complex<float> *x, std::int64_t incx,
                                  std::complex<float> *a, std::int64_t lda,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event her(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                                  uplo upper_lower, std::int64_t n, double alpha,
                                  const std::complex<double> *x, std::int64_t incx,
                                  std::complex<double> *a, std::int64_t lda,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event hpr(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                                  uplo upper_lower, std::int64_t n, float alpha,
                                  const std::complex<float> *x, std::int64_t incx,
                                  std::complex<float> *a,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event hpr(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                                  uplo upper_lower, std::int64_t n, double alpha,
                                  const std::complex<double> *x, std::int64_t incx,
                                  std::complex<double> *a,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event iamin(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, const float *x,
    std::int64_t incx, std::int64_t *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event iamin(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, const double *x,
    std::int64_t incx, std::int64_t *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event iamin(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
    const std::complex<float> *x, std::int64_t incx, std::int64_t *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event iamin(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
    const std::complex<double> *x, std::int64_t incx, std::int64_t *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event hpmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *a, const std::complex<float> *x,
    std::int64_t incx, std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event hpmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    std::complex<double> alpha, const std::complex<double> *a, const std::complex<double> *x,
    std::int64_t incx, std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event spmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    float alpha, const float *a, const float *x, std::int64_t incx, float beta, float *y,
    std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event spmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    double alpha, const double *a, const double *x, std::int64_t incx, double beta, double *y,
    std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event rotmg(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, float *d1, float *d2, float *x1, float y1,
    float *param, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event rotmg(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, double *d1, double *d2, double *x1,
    double y1, double *param, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event swap(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx,
    float *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event swap(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, double *x,
    std::int64_t incx, double *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event swap(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x,
    std::int64_t incx, std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event swap(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, std::complex<double> *x,
    std::int64_t incx, std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event geru(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
    const std::complex<float> *y, std::int64_t incy, std::complex<float> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event geru(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
    std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
    const std::complex<double> *y, std::int64_t incy, std::complex<double> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event nrm2(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
    const std::complex<float> *x, std::int64_t incx, float *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event nrm2(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
    const std::complex<double> *x, std::int64_t incx, double *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event nrm2(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, const float *x,
    std::int64_t incx, float *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event nrm2(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, const double *x,
    std::int64_t incx, double *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event gemmt(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose transa,
    transpose transb, std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
    const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gemmt(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose transa,
    transpose transb, std::int64_t n, std::int64_t k, double alpha, const double *a,
    std::int64_t lda, const double *b, std::int64_t ldb, double beta, double *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gemmt(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose transa,
    transpose transb, std::int64_t n, std::int64_t k, std::complex<float> alpha,
    const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b, std::int64_t ldb,
    std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gemmt(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose transa,
    transpose transb, std::int64_t n, std::int64_t k, std::complex<double> alpha,
    const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
    std::int64_t ldb, std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event gemm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa, transpose transb,
    std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
    const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gemm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa, transpose transb,
    std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double *a, std::int64_t lda,
    const double *b, std::int64_t ldb, double beta, double *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gemm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa, transpose transb,
    std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
    const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b, std::int64_t ldb,
    std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gemm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa, transpose transb,
    std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
    const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
    std::int64_t ldb, std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event syr2(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    float alpha, const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
    std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event syr2(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    double alpha, const double *x, std::int64_t incx, const double *y, std::int64_t incy, double *a,
    std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event ger(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                                  std::int64_t m, std::int64_t n, float alpha, const float *x,
                                  std::int64_t incx, const float *y, std::int64_t incy, float *a,
                                  std::int64_t lda,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event ger(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                                  std::int64_t m, std::int64_t n, double alpha, const double *x,
                                  std::int64_t incx, const double *y, std::int64_t incy, double *a,
                                  std::int64_t lda,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event trsm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, float alpha, const float *a,
    std::int64_t lda, float *b, std::int64_t ldb,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event trsm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, double alpha, const double *a,
    std::int64_t lda, double *b, std::int64_t ldb,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event trsm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
    const std::complex<float> *a, std::int64_t lda, std::complex<float> *b, std::int64_t ldb,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event trsm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
    const std::complex<double> *a, std::int64_t lda, std::complex<double> *b, std::int64_t ldb,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event dotu(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
    const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
    std::int64_t incy, std::complex<float> *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event dotu(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
    const std::complex<double> *x, std::int64_t incx, const std::complex<double> *y,
    std::int64_t incy, std::complex<double> *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event hemm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
    std::int64_t m, std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
    std::int64_t lda, const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
    std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event hemm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
    std::int64_t m, std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
    std::int64_t lda, const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
    std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event hpr2(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
    const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event hpr2(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
    const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event gbmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans, std::int64_t m,
    std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha, const float *a, std::int64_t lda,
    const float *x, std::int64_t incx, float beta, float *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gbmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans, std::int64_t m,
    std::int64_t n, std::int64_t kl, std::int64_t ku, double alpha, const double *a,
    std::int64_t lda, const double *x, std::int64_t incx, double beta, double *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gbmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans, std::int64_t m,
    std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
    const std::complex<float> *a, std::int64_t lda, const std::complex<float> *x, std::int64_t incx,
    std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event gbmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans, std::int64_t m,
    std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
    const std::complex<double> *a, std::int64_t lda, const std::complex<double> *x,
    std::int64_t incx, std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event tbmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, std::int64_t k, const float *a, std::int64_t lda, float *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event tbmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, std::int64_t k, const double *a, std::int64_t lda, double *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event tbmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, std::int64_t k, const std::complex<float> *a, std::int64_t lda,
    std::complex<float> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event tbmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, std::int64_t k, const std::complex<double> *a, std::int64_t lda,
    std::complex<double> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event symm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
    std::int64_t m, std::int64_t n, float alpha, const float *a, std::int64_t lda, const float *b,
    std::int64_t ldb, float beta, float *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event symm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
    std::int64_t m, std::int64_t n, double alpha, const double *a, std::int64_t lda,
    const double *b, std::int64_t ldb, double beta, double *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event symm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
    std::int64_t m, std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
    std::int64_t lda, const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
    std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event symm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
    std::int64_t m, std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
    std::int64_t lda, const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
    std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event dotc(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
    const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
    std::int64_t incy, std::complex<float> *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event dotc(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
    const std::complex<double> *x, std::int64_t incx, const std::complex<double> *y,
    std::int64_t incy, std::complex<double> *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event syr(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                                  uplo upper_lower, std::int64_t n, float alpha, const float *x,
                                  std::int64_t incx, float *a, std::int64_t lda,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event syr(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                                  uplo upper_lower, std::int64_t n, double alpha, const double *x,
                                  std::int64_t incx, double *a, std::int64_t lda,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event trmm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, float alpha, const float *a,
    std::int64_t lda, float *b, std::int64_t ldb,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event trmm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, double alpha, const double *a,
    std::int64_t lda, double *b, std::int64_t ldb,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event trmm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
    const std::complex<float> *a, std::int64_t lda, std::complex<float> *b, std::int64_t ldb,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event trmm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
    const std::complex<double> *a, std::int64_t lda, std::complex<double> *b, std::int64_t ldb,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event symv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    float alpha, const float *a, std::int64_t lda, const float *x, std::int64_t incx, float beta,
    float *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event symv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    double alpha, const double *a, std::int64_t lda, const double *x, std::int64_t incx,
    double beta, double *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event tpsv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, const float *a, float *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event tpsv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, const double *a, double *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event tpsv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, const std::complex<float> *a, std::complex<float> *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event tpsv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, const std::complex<double> *a, std::complex<double> *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event trsv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, const float *a, std::int64_t lda, float *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event trsv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, const double *a, std::int64_t lda, double *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event trsv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, const std::complex<float> *a, std::int64_t lda,
    std::complex<float> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event trsv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, const std::complex<double> *a, std::int64_t lda,
    std::complex<double> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event copy(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, const float *x,
    std::int64_t incx, float *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event copy(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, const double *x,
    std::int64_t incx, double *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event copy(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
    const std::complex<float> *x, std::int64_t incx, std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event copy(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
    const std::complex<double> *x, std::int64_t incx, std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event hemv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
    std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event hemv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
    std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event iamax(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, const float *x,
    std::int64_t incx, std::int64_t *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event iamax(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, const double *x,
    std::int64_t incx, std::int64_t *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event iamax(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
    const std::complex<float> *x, std::int64_t incx, std::int64_t *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event iamax(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
    const std::complex<double> *x, std::int64_t incx, std::int64_t *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event sbmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    std::int64_t k, float alpha, const float *a, std::int64_t lda, const float *x,
    std::int64_t incx, float beta, float *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event sbmv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    std::int64_t k, double alpha, const double *a, std::int64_t lda, const double *x,
    std::int64_t incx, double beta, double *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event asum(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
    const std::complex<float> *x, std::int64_t incx, float *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event asum(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
    const std::complex<double> *x, std::int64_t incx, double *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event asum(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, const float *x,
    std::int64_t incx, float *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event asum(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, const double *x,
    std::int64_t incx, double *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event tbsv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, std::int64_t k, const float *a, std::int64_t lda, float *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event tbsv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, std::int64_t k, const double *a, std::int64_t lda, double *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event tbsv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, std::int64_t k, const std::complex<float> *a, std::int64_t lda,
    std::complex<float> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event tbsv(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    diag unit_diag, std::int64_t n, std::int64_t k, const std::complex<double> *a, std::int64_t lda,
    std::complex<double> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event spr2(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    float alpha, const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event spr2(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
    double alpha, const double *x, std::int64_t incx, const double *y, std::int64_t incy, double *a,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event rotm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx,
    float *y, std::int64_t incy, float *param,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event rotm(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, double *x,
    std::int64_t incx, double *y, std::int64_t incy, double *param,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event dot(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                                  std::int64_t n, const float *x, std::int64_t incx, const float *y,
                                  std::int64_t incy, float *result,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event dot(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                                  std::int64_t n, const double *x, std::int64_t incx,
                                  const double *y, std::int64_t incy, double *result,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event dot(oneapi::mkl::device libkey, cl::sycl::queue &queue,
                                  std::int64_t n, const float *x, std::int64_t incx, const float *y,
                                  std::int64_t incy, double *result,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event sdsdot(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, float sb, const float *x,
    std::int64_t incx, const float *y, std::int64_t incy, float *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event her2k(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    std::int64_t n, std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
    std::int64_t lda, const std::complex<float> *b, std::int64_t ldb, float beta,
    std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event her2k(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
    std::int64_t n, std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
    std::int64_t lda, const std::complex<double> *b, std::int64_t ldb, double beta,
    std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});

ONEMKL_EXPORT cl::sycl::event rotg(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, float *a, float *b, float *c, float *s,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event rotg(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, double *a, double *b, double *c, double *s,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event rotg(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::complex<float> *a,
    std::complex<float> *b, float *c, std::complex<float> *s,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
ONEMKL_EXPORT cl::sycl::event rotg(
    oneapi::mkl::device libkey, cl::sycl::queue &queue, std::complex<double> *a,
    std::complex<double> *b, double *c, std::complex<double> *s,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {});
