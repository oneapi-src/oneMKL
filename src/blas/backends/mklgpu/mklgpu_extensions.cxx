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

void gemm_bias(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
               oneapi::mkl::offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
               float alpha, cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda, int8_t ao,
               cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb, uint8_t bo, float beta,
               cl::sycl::buffer<int32_t, 1> &c, std::int64_t ldc,
               cl::sycl::buffer<int32_t, 1> &co) {
#ifdef COLUMN_MAJOR
    ::oneapi::mkl::gpu::gemm_s8u8s32_sycl(&queue, MAJOR, ::mkl::cblas_convert(transa),
                                          ::mkl::cblas_convert(transb),
                                          ::mkl::cblas_convert(offsetc), m, n, k, alpha, &a, lda,
                                          ao, &b, ldb, bo, beta, &c, ldc, &co);
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_bias", "for row_major layout");
#endif
}

void gemmt(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, float alpha,
           cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
           std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::sgemmt(queue, MAJOR, ::mkl::cblas_convert(upper_lower),
                               ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k,
                               alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, double alpha,
           cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
           std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::dgemmt(queue, MAJOR, ::mkl::cblas_convert(upper_lower),
                               ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k,
                               alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
           std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc) {
    ::oneapi::mkl::gpu::zgemmt(queue, MAJOR, ::mkl::cblas_convert(upper_lower),
                               ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k,
                               alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::cgemmt(queue, MAJOR, ::mkl::cblas_convert(upper_lower),
                               ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k,
                               alpha, a, lda, b, ldb, beta, c, ldc);
}

// USM APIs

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
                      const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sgemmt_sycl(
        &queue, MAJOR, ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
        ::mkl::cblas_convert(transb), n, k, alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, double alpha, const double *a,
                      std::int64_t lda, const double *b, std::int64_t ldb, double beta, double *c,
                      std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dgemmt_sycl(
        &queue, MAJOR, ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
        ::mkl::cblas_convert(transb), n, k, alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, std::complex<float> alpha,
                      const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                      std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                      std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgemmt_sycl(
        &queue, MAJOR, ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
        ::mkl::cblas_convert(transb), n, k, alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, std::complex<double> alpha,
                      const std::complex<double> *a, std::int64_t lda,
                      const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                      std::complex<double> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgemmt_sycl(
        &queue, MAJOR, ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
        ::mkl::cblas_convert(transb), n, k, alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}
