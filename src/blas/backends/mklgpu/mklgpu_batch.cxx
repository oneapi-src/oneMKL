/*******************************************************************************
* Copyright 2020 Intel Corporation
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

void gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b, float beta,
                cl::sycl::buffer<float, 1> &c, std::int64_t ldc, std::int64_t stride_c,
                std::int64_t batch_size) {
    ::oneapi::mkl::gpu::sgemm_batch(queue, MAJOR, ::mkl::cblas_convert(transa),
                                    ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, stride_a,
                                    b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

void gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, double beta, cl::sycl::buffer<double, 1> &c,
                std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
    ::oneapi::mkl::gpu::dgemm_batch(queue, MAJOR, ::mkl::cblas_convert(transa),
                                    ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, stride_a,
                                    b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

void gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::complex<float> beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    ::oneapi::mkl::gpu::cgemm_batch(queue, MAJOR, ::mkl::cblas_convert(transa),
                                    ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, stride_a,
                                    b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

void gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::complex<double> beta,
                cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    ::oneapi::mkl::gpu::zgemm_batch(queue, MAJOR, ::mkl::cblas_convert(transa),
                                    ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, stride_a,
                                    b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

void trsm_batch(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    ::oneapi::mkl::gpu::strsm_batch(queue, MAJOR, ::mkl::cblas_convert(left_right),
                                    ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(trans),
                                    ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, stride_a,
                                    b, ldb, stride_b, batch_size);
}

void trsm_batch(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    ::oneapi::mkl::gpu::dtrsm_batch(queue, MAJOR, ::mkl::cblas_convert(left_right),
                                    ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(trans),
                                    ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, stride_a,
                                    b, ldb, stride_b, batch_size);
}

void trsm_batch(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                std::int64_t n, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    ::oneapi::mkl::gpu::ctrsm_batch(queue, MAJOR, ::mkl::cblas_convert(left_right),
                                    ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(trans),
                                    ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, stride_a,
                                    b, ldb, stride_b, batch_size);
}

void trsm_batch(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                std::int64_t n, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    ::oneapi::mkl::gpu::ztrsm_batch(queue, MAJOR, ::mkl::cblas_convert(left_right),
                                    ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(trans),
                                    ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, stride_a,
                                    b, ldb, stride_b, batch_size);
}

// USM APIs

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                           const float *a, std::int64_t lda, std::int64_t stride_a, const float *b,
                           std::int64_t ldb, std::int64_t stride_b, float beta, float *c,
                           std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sgemm_batch_sycl(&queue, MAJOR, ::mkl::cblas_convert(transa),
                                                ::mkl::cblas_convert(transb), m, n, k, alpha, a,
                                                lda, stride_a, b, ldb, stride_b, beta, c, ldc,
                                                stride_c, batch_size, dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                           const double *a, std::int64_t lda, std::int64_t stride_a,
                           const double *b, std::int64_t ldb, std::int64_t stride_b, double beta,
                           double *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dgemm_batch_sycl(&queue, MAJOR, ::mkl::cblas_convert(transa),
                                                ::mkl::cblas_convert(transb), m, n, k, alpha, a,
                                                lda, stride_a, b, ldb, stride_b, beta, c, ldc,
                                                stride_c, batch_size, dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<float> alpha, const std::complex<float> *a,
                           std::int64_t lda, std::int64_t stride_a, const std::complex<float> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                           std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgemm_batch_sycl(&queue, MAJOR, ::mkl::cblas_convert(transa),
                                                ::mkl::cblas_convert(transb), m, n, k, alpha, a,
                                                lda, stride_a, b, ldb, stride_b, beta, c, ldc,
                                                stride_c, batch_size, dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<double> alpha, const std::complex<double> *a,
                           std::int64_t lda, std::int64_t stride_a, const std::complex<double> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                           std::complex<double> *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgemm_batch_sycl(&queue, MAJOR, ::mkl::cblas_convert(transa),
                                                ::mkl::cblas_convert(transb), m, n, k, alpha, a,
                                                lda, stride_a, b, ldb, stride_b, beta, c, ldc,
                                                stride_c, batch_size, dependencies);
}

cl::sycl::event *coalesce_events(cl::sycl::queue &queue, std::vector<cl::sycl::event *> &prereqs) {
#ifdef _WIN64
    for (std::int64_t i = 0; i < prereqs.size(); i++)
        prereqs[i]->wait();
    return new cl::sycl::event();
#else
    if (prereqs.size() > 0) {
        return new cl::sycl::event(queue.submit([&](cl::sycl::handler &cgh) {
            for (std::int64_t i = 0; i < prereqs.size(); i++)
                cgh.depends_on(*prereqs[i]);
            cgh.single_task<class MAJOR>([]() {});
        }));
    }
    else
        return new cl::sycl::event();
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k, float *alpha,
                           const float **a, std::int64_t *lda, const float **b, std::int64_t *ldb,
                           float *beta, float **c, std::int64_t *ldc, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    std::vector<cl::sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    std::int64_t total_group_size = 0;
    for (std::int64_t i = 0; i < group_count; i++) {
        cl::sycl::event *gemm_batch_event =
            new cl::sycl::event(::oneapi::mkl::gpu::sgemm_batch_sycl(
                &queue, MAJOR, ::mkl::cblas_convert(transa[i]),
                ::mkl::cblas_convert(transb[i]), m[i], n[i], k[i], alpha[i], a, lda[i], b, ldb[i],
                beta[i], c, ldc[i], total_group_size, group_size[i], dependencies));
        coalesced_events.push_back(gemm_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k, double *alpha,
                           const double **a, std::int64_t *lda, const double **b, std::int64_t *ldb,
                           double *beta, double **c, std::int64_t *ldc, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    std::vector<cl::sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    std::int64_t total_group_size = 0;
    for (std::int64_t i = 0; i < group_count; i++) {
        cl::sycl::event *gemm_batch_event =
            new cl::sycl::event(::oneapi::mkl::gpu::dgemm_batch_sycl(
                &queue, MAJOR, ::mkl::cblas_convert(transa[i]),
                ::mkl::cblas_convert(transb[i]), m[i], n[i], k[i], alpha[i], a, lda[i], b, ldb[i],
                beta[i], c, ldc[i], total_group_size, group_size[i], dependencies));
        coalesced_events.push_back(gemm_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k,
                           std::complex<float> *alpha, const std::complex<float> **a,
                           std::int64_t *lda, const std::complex<float> **b, std::int64_t *ldb,
                           std::complex<float> *beta, std::complex<float> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    std::vector<cl::sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    std::int64_t total_group_size = 0;
    for (std::int64_t i = 0; i < group_count; i++) {
        cl::sycl::event *gemm_batch_event =
            new cl::sycl::event(::oneapi::mkl::gpu::cgemm_batch_sycl(
                &queue, MAJOR, ::mkl::cblas_convert(transa[i]),
                ::mkl::cblas_convert(transb[i]), m[i], n[i], k[i], alpha[i], a, lda[i], b, ldb[i],
                beta[i], c, ldc[i], total_group_size, group_size[i], dependencies));
        coalesced_events.push_back(gemm_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k,
                           std::complex<double> *alpha, const std::complex<double> **a,
                           std::int64_t *lda, const std::complex<double> **b, std::int64_t *ldb,
                           std::complex<double> *beta, std::complex<double> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    std::vector<cl::sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    std::int64_t total_group_size = 0;
    for (std::int64_t i = 0; i < group_count; i++) {
        cl::sycl::event *gemm_batch_event =
            new cl::sycl::event(::oneapi::mkl::gpu::zgemm_batch_sycl(
                &queue, MAJOR, ::mkl::cblas_convert(transa[i]),
                ::mkl::cblas_convert(transb[i]), m[i], n[i], k[i], alpha[i], a, lda[i], b, ldb[i],
                beta[i], c, ldc[i], total_group_size, group_size[i], dependencies));
        coalesced_events.push_back(gemm_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, float *alpha, const float **x,
                           std::int64_t *incx, float **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::saxpy_batch(queue, n, alpha, x, incx, y, incy, group_count,
                                           group_size, dependencies);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, double *alpha, const double **x,
                           std::int64_t *incx, double **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::daxpy_batch(queue, n, alpha, x, incx, y, incy, group_count,
                                           group_size, dependencies);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, std::complex<float> *alpha,
                           const std::complex<float> **x, std::int64_t *incx,
                           std::complex<float> **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::caxpy_batch(queue, n, alpha, x, incx, y, incy, group_count,
                                           group_size, dependencies);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, std::complex<double> *alpha,
                           const std::complex<double> **x, std::int64_t *incx,
                           std::complex<double> **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zaxpy_batch(queue, n, alpha, x, incx, y, incy, group_count,
                                           group_size, dependencies);
}

