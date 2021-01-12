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

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<float, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, float beta, cl::sycl::buffer<float, 1> &c,
                std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<double, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, double beta,
                cl::sycl::buffer<double, 1> &c, std::int64_t ldc, std::int64_t stride_c,
                std::int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                cl::sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

// USM APIs

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, float *alpha, const float **x,
                           int64_t *incx, float **y, int64_t *incy, int64_t group_count,
                           int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, double *alpha, const double **x,
                           int64_t *incx, double **y, int64_t *incy, int64_t group_count,
                           int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, std::complex<float> *alpha,
                           const std::complex<float> **x, int64_t *incx, std::complex<float> **y,
                           int64_t *incy, int64_t group_count, int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, std::complex<double> *alpha,
                           const std::complex<double> **x, int64_t *incx, std::complex<double> **y,
                           int64_t *incy, int64_t group_count, int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k, float *alpha,
                           const float **a, std::int64_t *lda, const float **b, std::int64_t *ldb,
                           float *beta, float **c, std::int64_t *ldc, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k, double *alpha,
                           const double **a, std::int64_t *lda, const double **b, std::int64_t *ldb,
                           double *beta, double **c, std::int64_t *ldc, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k,
                           std::complex<float> *alpha, const std::complex<float> **a,
                           std::int64_t *lda, const std::complex<float> **b, std::int64_t *ldb,
                           std::complex<float> *beta, std::complex<float> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k,
                           std::complex<double> *alpha, const std::complex<double> **a,
                           std::int64_t *lda, const std::complex<double> **b, std::int64_t *ldb,
                           std::complex<double> *beta, std::complex<double> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                           const float *a, std::int64_t lda, std::int64_t stride_a, const float *b,
                           std::int64_t ldb, std::int64_t stride_b, float beta, float *c,
                           std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                           const double *a, std::int64_t lda, std::int64_t stride_a,
                           const double *b, std::int64_t ldb, std::int64_t stride_b, double beta,
                           double *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<float> alpha, const std::complex<float> *a,
                           std::int64_t lda, std::int64_t stride_a, const std::complex<float> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                           std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<double> alpha, const std::complex<double> *a,
                           std::int64_t lda, std::int64_t stride_a, const std::complex<double> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                           std::complex<double> *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}
