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

void copy_batch(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
                int64_t stridex, cl::sycl::buffer<float, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

void copy_batch(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
                int64_t stridex, cl::sycl::buffer<double, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

void copy_batch(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
                int64_t incx, int64_t stridex, cl::sycl::buffer<std::complex<float>, 1> &y,
                int64_t incy, int64_t stridey, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

void copy_batch(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
                int64_t incx, int64_t stridex, cl::sycl::buffer<std::complex<double>, 1> &y,
                int64_t incy, int64_t stridey, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

void axpy_batch(cl::sycl::queue &queue, int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
                int64_t incx, int64_t stridex, cl::sycl::buffer<float, 1> &y, int64_t incy,
                int64_t stridey, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

void axpy_batch(cl::sycl::queue &queue, int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
                int64_t incx, int64_t stridex, cl::sycl::buffer<double, 1> &y, int64_t incy,
                int64_t stridey, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

void axpy_batch(cl::sycl::queue &queue, int64_t n, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, int64_t stridex,
                cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

void axpy_batch(cl::sycl::queue &queue, int64_t n, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, int64_t stridex,
                cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

void gemv_batch(cl::sycl::queue &queue, transpose transa, int64_t m, int64_t n, float alpha,
                cl::sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a,
                cl::sycl::buffer<float, 1> &x, int64_t incx, int64_t stride_x, float beta,
                cl::sycl::buffer<float, 1> &y, int64_t incy, int64_t stride_y, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

void gemv_batch(cl::sycl::queue &queue, transpose transa, int64_t m, int64_t n, double alpha,
                cl::sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                cl::sycl::buffer<double, 1> &x, int64_t incx, int64_t stride_x, double beta,
                cl::sycl::buffer<double, 1> &y, int64_t incy, int64_t stride_y,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

void gemv_batch(cl::sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
                int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
                int64_t stride_x, std::complex<float> beta,
                cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy, int64_t stride_y,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

void gemv_batch(cl::sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                int64_t lda, int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &x,
                int64_t incx, int64_t stride_x, std::complex<double> beta,
                cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy, int64_t stride_y,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

void dgmm_batch(cl::sycl::queue &queue, side left_right, int64_t m, int64_t n,
                cl::sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a,
                cl::sycl::buffer<float, 1> &x, int64_t incx, int64_t stride_x,
                cl::sycl::buffer<float, 1> &c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

void dgmm_batch(cl::sycl::queue &queue, side left_right, int64_t m, int64_t n,
                cl::sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                cl::sycl::buffer<double, 1> &x, int64_t incx, int64_t stride_x,
                cl::sycl::buffer<double, 1> &c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

void dgmm_batch(cl::sycl::queue &queue, side left_right, int64_t m, int64_t n,
                cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda, int64_t stride_a,
                cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, int64_t stride_x,
                cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

void dgmm_batch(cl::sycl::queue &queue, side left_right, int64_t m, int64_t n,
                cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda, int64_t stride_a,
                cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, int64_t stride_x,
                cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
                int64_t stride_a, cl::sycl::buffer<float, 1> &b, int64_t ldb, int64_t stride_b,
                float beta, cl::sycl::buffer<float, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
                int64_t stride_a, cl::sycl::buffer<double, 1> &b, int64_t ldb, int64_t stride_b,
                double beta, cl::sycl::buffer<double, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                int64_t lda, int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                int64_t ldb, int64_t stride_b, std::complex<float> beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                int64_t lda, int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                int64_t ldb, int64_t stride_b, std::complex<double> beta,
                cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, half alpha, cl::sycl::buffer<half, 1> &a, int64_t lda, int64_t stride_a,
                cl::sycl::buffer<half, 1> &b, int64_t ldb, int64_t stride_b, half beta,
                cl::sycl::buffer<half, 1> &c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                int64_t lda, int64_t stride_a, cl::sycl::buffer<float, 1> &b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                int64_t lda, int64_t stride_a, cl::sycl::buffer<double, 1> &b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda, int64_t stride_a,
                cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda, int64_t stride_a,
                cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

void syrk_batch(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a,
                float beta, cl::sycl::buffer<float, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

void syrk_batch(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                double beta, cl::sycl::buffer<double, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

void syrk_batch(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
                int64_t stride_a, std::complex<float> beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

void syrk_batch(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                int64_t lda, int64_t stride_a, std::complex<double> beta,
                cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

// USM APIs

cl::sycl::event copy_batch(cl::sycl::queue &queue, int64_t *n, const float **x, int64_t *incx,
                           float **y, int64_t *incy, int64_t group_count, int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

cl::sycl::event copy_batch(cl::sycl::queue &queue, int64_t *n, const double **x, int64_t *incx,
                           double **y, int64_t *incy, int64_t group_count, int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

cl::sycl::event copy_batch(cl::sycl::queue &queue, int64_t *n, const std::complex<float> **x,
                           int64_t *incx, std::complex<float> **y, int64_t *incy,
                           int64_t group_count, int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

cl::sycl::event copy_batch(cl::sycl::queue &queue, int64_t *n, const std::complex<double> **x,
                           int64_t *incx, std::complex<double> **y, int64_t *incy,
                           int64_t group_count, int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

cl::sycl::event copy_batch(cl::sycl::queue &queue, int64_t n, const float *x, int64_t incx,
                           std::int64_t stridex, float *y, int64_t incy, std::int64_t stridey,
                           std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

cl::sycl::event copy_batch(cl::sycl::queue &queue, int64_t n, const double *x, int64_t incx,
                           std::int64_t stridex, double *y, int64_t incy, std::int64_t stridey,
                           std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

cl::sycl::event copy_batch(cl::sycl::queue &queue, int64_t n, const std::complex<float> *x,
                           int64_t incx, std::int64_t stridex, std::complex<float> *y, int64_t incy,
                           std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

cl::sycl::event copy_batch(cl::sycl::queue &queue, int64_t n, const std::complex<double> *x,
                           int64_t incx, std::int64_t stridex, std::complex<double> *y,
                           int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, float *alpha, const float **x,
                           int64_t *incx, float **y, int64_t *incy, int64_t group_count,
                           int64_t *group_size, const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, double *alpha, const double **x,
                           int64_t *incx, double **y, int64_t *incy, int64_t group_count,
                           int64_t *group_size, const std::vector<cl::sycl::event> &dependencies) {
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
                           const std::vector<cl::sycl::event> &dependencies) {
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
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t n, float alpha, const float *x,
                           int64_t incx, int64_t stridex, float *y, int64_t incy, int64_t stridey,
                           int64_t batch_size, const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t n, double alpha, const double *x,
                           int64_t incx, int64_t stridex, double *y, int64_t incy, int64_t stridey,
                           int64_t batch_size, const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t n, std::complex<float> alpha,
                           const std::complex<float> *x, int64_t incx, int64_t stridex,
                           std::complex<float> *y, int64_t incy, int64_t stridey,
                           int64_t batch_size, const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t n, std::complex<double> alpha,
                           const std::complex<double> *x, int64_t incx, int64_t stridex,
                           std::complex<double> *y, int64_t incy, int64_t stridey,
                           int64_t batch_size, const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

cl::sycl::event gemv_batch(cl::sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                           float alpha, const float *a, int64_t lda, int64_t stride_a,
                           const float *x, int64_t incx, int64_t stride_x, float beta, float *y,
                           int64_t incy, int64_t stride_y, int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

cl::sycl::event gemv_batch(cl::sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                           double alpha, const double *a, int64_t lda, int64_t stride_a,
                           const double *x, int64_t incx, int64_t stride_x, double beta, double *y,
                           int64_t incy, int64_t stride_y, int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

cl::sycl::event gemv_batch(cl::sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                           std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                           int64_t stride_a, const std::complex<float> *x, int64_t incx,
                           int64_t stride_x, std::complex<float> beta, std::complex<float> *y,
                           int64_t incy, int64_t stride_y, int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

cl::sycl::event gemv_batch(cl::sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                           std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                           int64_t stride_a, const std::complex<double> *x, int64_t incx,
                           int64_t stride_x, std::complex<double> beta, std::complex<double> *y,
                           int64_t incy, int64_t stride_y, int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

cl::sycl::event gemv_batch(cl::sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n,
                           float *alpha, const float **a, int64_t *lda, const float **x,
                           int64_t *incx, float *beta, float **y, int64_t *incy,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

cl::sycl::event gemv_batch(cl::sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n,
                           double *alpha, const double **a, int64_t *lda, const double **x,
                           int64_t *incx, double *beta, double **y, int64_t *incy,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

cl::sycl::event gemv_batch(cl::sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n,
                           std::complex<float> *alpha, const std::complex<float> **a, int64_t *lda,
                           const std::complex<float> **x, int64_t *incx, std::complex<float> *beta,
                           std::complex<float> **y, int64_t *incy, int64_t group_count,
                           int64_t *groupsize, const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

cl::sycl::event gemv_batch(cl::sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n,
                           std::complex<double> *alpha, const std::complex<double> **a,
                           int64_t *lda, const std::complex<double> **x, int64_t *incx,
                           std::complex<double> *beta, std::complex<double> **y, int64_t *incy,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

cl::sycl::event dgmm_batch(cl::sycl::queue &queue, side left_right, int64_t m, int64_t n,
                           const float *a, int64_t lda, int64_t stride_a, const float *x,
                           int64_t incx, int64_t stride_x, float *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

cl::sycl::event dgmm_batch(cl::sycl::queue &queue, side left_right, int64_t m, int64_t n,
                           const double *a, int64_t lda, int64_t stride_a, const double *x,
                           int64_t incx, int64_t stride_x, double *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

cl::sycl::event dgmm_batch(cl::sycl::queue &queue, side left_right, int64_t m, int64_t n,
                           const std::complex<float> *a, int64_t lda, int64_t stride_a,
                           const std::complex<float> *x, int64_t incx, int64_t stride_x,
                           std::complex<float> *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

cl::sycl::event dgmm_batch(cl::sycl::queue &queue, side left_right, int64_t m, int64_t n,
                           const std::complex<double> *a, int64_t lda, int64_t stride_a,
                           const std::complex<double> *x, int64_t incx, int64_t stride_x,
                           std::complex<double> *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

cl::sycl::event dgmm_batch(cl::sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                           const float **a, int64_t *lda, const float **x, int64_t *incx, float **c,
                           int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

cl::sycl::event dgmm_batch(cl::sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                           const double **a, int64_t *lda, const double **x, int64_t *incx,
                           double **c, int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

cl::sycl::event dgmm_batch(cl::sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                           const std::complex<float> **a, int64_t *lda,
                           const std::complex<float> **x, int64_t *incx, std::complex<float> **c,
                           int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

cl::sycl::event dgmm_batch(cl::sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                           const std::complex<double> **a, int64_t *lda,
                           const std::complex<double> **x, int64_t *incx, std::complex<double> **c,
                           int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,
                           int64_t *n, int64_t *k, float *alpha, const float **a, int64_t *lda,
                           const float **b, int64_t *ldb, float *beta, float **c, int64_t *ldc,
                           int64_t group_count, int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,
                           int64_t *n, int64_t *k, double *alpha, const double **a, int64_t *lda,
                           const double **b, int64_t *ldb, double *beta, double **c, int64_t *ldc,
                           int64_t group_count, int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,
                           int64_t *n, int64_t *k, std::complex<float> *alpha,
                           const std::complex<float> **a, int64_t *lda,
                           const std::complex<float> **b, int64_t *ldb, std::complex<float> *beta,
                           std::complex<float> **c, int64_t *ldc, int64_t group_count,
                           int64_t *group_size, const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,
                           int64_t *n, int64_t *k, std::complex<double> *alpha,
                           const std::complex<double> **a, int64_t *lda,
                           const std::complex<double> **b, int64_t *ldb, std::complex<double> *beta,
                           std::complex<double> **c, int64_t *ldc, int64_t group_count,
                           int64_t *group_size, const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,
                           int64_t *n, int64_t *k, half *alpha, const half **a, int64_t *lda,
                           const half **b, int64_t *ldb, half *beta, half **c, int64_t *ldc,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                           int64_t n, int64_t k, float alpha, const float *a, int64_t lda,
                           int64_t stride_a, const float *b, int64_t ldb, int64_t stride_b,
                           float beta, float *c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                           int64_t n, int64_t k, double alpha, const double *a, int64_t lda,
                           int64_t stride_a, const double *b, int64_t ldb, int64_t stride_b,
                           double beta, double *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                           int64_t n, int64_t k, std::complex<float> alpha,
                           const std::complex<float> *a, int64_t lda, int64_t stride_a,
                           const std::complex<float> *b, int64_t ldb, int64_t stride_b,
                           std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                           int64_t stride_c, int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                           int64_t n, int64_t k, std::complex<double> alpha,
                           const std::complex<double> *a, int64_t lda, int64_t stride_a,
                           const std::complex<double> *b, int64_t ldb, int64_t stride_b,
                           std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                           int64_t stride_c, int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                           int64_t n, int64_t k, half alpha, const half *a, int64_t lda,
                           int64_t stride_a, const half *b, int64_t ldb, int64_t stride_b,
                           half beta, half *c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

cl::sycl::event trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, int64_t m, int64_t n, float alpha,
                           const float *a, int64_t lda, int64_t stride_a, float *b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

cl::sycl::event trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, int64_t m, int64_t n, double alpha,
                           const double *a, int64_t lda, int64_t stride_a, double *b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

cl::sycl::event trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, int64_t m, int64_t n,
                           std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                           int64_t stride_a, std::complex<float> *b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

cl::sycl::event trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, int64_t m, int64_t n,
                           std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                           int64_t stride_a, std::complex<double> *b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

cl::sycl::event trsm_batch(cl::sycl::queue &queue, side *left_right, uplo *upper_lower,
                           transpose *trans, diag *unit_diag, int64_t *m, int64_t *n, float *alpha,
                           const float **a, int64_t *lda, float **b, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

cl::sycl::event trsm_batch(cl::sycl::queue &queue, side *left_right, uplo *upper_lower,
                           transpose *trans, diag *unit_diag, int64_t *m, int64_t *n, double *alpha,
                           const double **a, int64_t *lda, double **b, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

cl::sycl::event trsm_batch(cl::sycl::queue &queue, side *left_right, uplo *upper_lower,
                           transpose *trans, diag *unit_diag, int64_t *m, int64_t *n,
                           std::complex<float> *alpha, const std::complex<float> **a, int64_t *lda,
                           std::complex<float> **b, int64_t *ldb, int64_t group_count,
                           int64_t *groupsize, const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

cl::sycl::event trsm_batch(cl::sycl::queue &queue, side *left_right, uplo *upper_lower,
                           transpose *trans, diag *unit_diag, int64_t *m, int64_t *n,
                           std::complex<double> *alpha, const std::complex<double> **a,
                           int64_t *lda, std::complex<double> **b, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

cl::sycl::event syrk_batch(cl::sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                           int64_t *k, float *alpha, const float **a, int64_t *lda, float *beta,
                           float **c, int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

cl::sycl::event syrk_batch(cl::sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                           int64_t *k, double *alpha, const double **a, int64_t *lda, double *beta,
                           double **c, int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

cl::sycl::event syrk_batch(cl::sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                           int64_t *k, std::complex<float> *alpha, const std::complex<float> **a,
                           int64_t *lda, std::complex<float> *beta, std::complex<float> **c,
                           int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

cl::sycl::event syrk_batch(cl::sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                           int64_t *k, std::complex<double> *alpha, const std::complex<double> **a,
                           int64_t *lda, std::complex<double> *beta, std::complex<double> **c,
                           int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

cl::sycl::event syrk_batch(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                           int64_t k, float alpha, const float *a, int64_t lda, int64_t stride_a,
                           float beta, float *c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

cl::sycl::event syrk_batch(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                           int64_t k, double alpha, const double *a, int64_t lda, int64_t stride_a,
                           double beta, double *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

cl::sycl::event syrk_batch(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                           int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                           int64_t lda, int64_t stride_a, std::complex<float> beta,
                           std::complex<float> *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

cl::sycl::event syrk_batch(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                           int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                           int64_t lda, int64_t stride_a, std::complex<double> beta,
                           std::complex<double> *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}
