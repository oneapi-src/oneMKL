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

void copy_batch(sycl::queue &queue, int64_t n, sycl::buffer<float, 1> &x, int64_t incx,
                std::int64_t stridex, sycl::buffer<float, 1> &y, int64_t incy,
                std::int64_t stridey, std::int64_t batch_size) {
    ::oneapi::mkl::gpu::scopy_batch_sycl(&queue, n, &x, incx, stridex, &y, incy, stridey,
                                         batch_size);
}

void copy_batch(sycl::queue &queue, int64_t n, sycl::buffer<double, 1> &x, int64_t incx,
                std::int64_t stridex, sycl::buffer<double, 1> &y, int64_t incy,
                std::int64_t stridey, std::int64_t batch_size) {
    ::oneapi::mkl::gpu::dcopy_batch_sycl(&queue, n, &x, incx, stridex, &y, incy, stridey,
                                         batch_size);
}

void copy_batch(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<float>, 1> &x,
                int64_t incx, std::int64_t stridex, sycl::buffer<std::complex<float>, 1> &y,
                int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
    ::oneapi::mkl::gpu::ccopy_batch_sycl(&queue, n, &x, incx, stridex, &y, incy, stridey,
                                         batch_size);
}

void copy_batch(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<double>, 1> &x,
                int64_t incx, std::int64_t stridex, sycl::buffer<std::complex<double>, 1> &y,
                int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
    ::oneapi::mkl::gpu::zcopy_batch_sycl(&queue, n, &x, incx, stridex, &y, incy, stridey,
                                         batch_size);
}

void axpy_batch(sycl::queue &queue, int64_t n, double alpha, sycl::buffer<double, 1> &x,
                int64_t incx, int64_t stridex, sycl::buffer<double, 1> &y, int64_t incy,
                int64_t stridey, int64_t batch_size) {
    ::oneapi::mkl::gpu::daxpy_batch(queue, n, alpha, x, incx, stridex, y, incy, stridey,
                                    batch_size);
}

void axpy_batch(sycl::queue &queue, int64_t n, float alpha, sycl::buffer<float, 1> &x,
                int64_t incx, int64_t stridex, sycl::buffer<float, 1> &y, int64_t incy,
                int64_t stridey, int64_t batch_size) {
    ::oneapi::mkl::gpu::saxpy_batch(queue, n, alpha, x, incx, stridex, y, incy, stridey,
                                    batch_size);
}

void axpy_batch(sycl::queue &queue, int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1> &x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<float>, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    ::oneapi::mkl::gpu::caxpy_batch(queue, n, alpha, x, incx, stridex, y, incy, stridey,
                                    batch_size);
}

void axpy_batch(sycl::queue &queue, int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1> &x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<double>, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    ::oneapi::mkl::gpu::zaxpy_batch(queue, n, alpha, x, incx, stridex, y, incy, stridey,
                                    batch_size);
}

void gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n, float alpha,
                sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<float, 1> &x, int64_t incx, int64_t stride_x, float beta,
                sycl::buffer<float, 1> &y, int64_t incy, int64_t stride_y, int64_t batch_size) {
    ::oneapi::mkl::gpu::sgemv_batch_sycl(&queue, MAJOR, mkl_convert(transa), m, n, alpha, &a, lda,
                                         stride_a, &x, incx, stride_x, beta, &y, incy, stride_y,
                                         batch_size);
}

void gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n, double alpha,
                sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<double, 1> &x, int64_t incx, int64_t stride_x, double beta,
                sycl::buffer<double, 1> &y, int64_t incy, int64_t stride_y,
                int64_t batch_size) {
    ::oneapi::mkl::gpu::dgemv_batch_sycl(&queue, MAJOR, mkl_convert(transa), m, n, alpha, &a, lda,
                                         stride_a, &x, incx, stride_x, beta, &y, incy, stride_y,
                                         batch_size);
}

void gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
                int64_t stride_x, std::complex<float> beta,
                sycl::buffer<std::complex<float>, 1> &y, int64_t incy, int64_t stride_y,
                int64_t batch_size) {
    ::oneapi::mkl::gpu::cgemv_batch_sycl(&queue, MAJOR, mkl_convert(transa), m, n, alpha, &a, lda,
                                         stride_a, &x, incx, stride_x, beta, &y, incy, stride_y,
                                         batch_size);
}

void gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
                int64_t lda, int64_t stride_a, sycl::buffer<std::complex<double>, 1> &x,
                int64_t incx, int64_t stride_x, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1> &y, int64_t incy, int64_t stride_y,
                int64_t batch_size) {
    ::oneapi::mkl::gpu::zgemv_batch_sycl(&queue, MAJOR, mkl_convert(transa), m, n, alpha, &a, lda,
                                         stride_a, &x, incx, stride_x, beta, &y, incy, stride_y,
                                         batch_size);
}

void dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<float, 1> &x, int64_t incx, int64_t stride_x,
                sycl::buffer<float, 1> &c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    ::oneapi::mkl::gpu::sdgmm_batch_sycl(&queue, MAJOR, mkl_convert(left_right), m, n, &a, lda,
                                         stride_a, &x, incx, stride_x, &c, ldc, stride_c,
                                         batch_size);
}

void dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<double, 1> &x, int64_t incx, int64_t stride_x,
                sycl::buffer<double, 1> &c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    ::oneapi::mkl::gpu::ddgmm_batch_sycl(&queue, MAJOR, mkl_convert(left_right), m, n, &a, lda,
                                         stride_a, &x, incx, stride_x, &c, ldc, stride_c,
                                         batch_size);
}

void dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<std::complex<float>, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<float>, 1> &x, int64_t incx, int64_t stride_x,
                sycl::buffer<std::complex<float>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    ::oneapi::mkl::gpu::cdgmm_batch_sycl(&queue, MAJOR, mkl_convert(left_right), m, n, &a, lda,
                                         stride_a, &x, incx, stride_x, &c, ldc, stride_c,
                                         batch_size);
}

void dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<std::complex<double>, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<double>, 1> &x, int64_t incx, int64_t stride_x,
                sycl::buffer<std::complex<double>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    ::oneapi::mkl::gpu::zdgmm_batch_sycl(&queue, MAJOR, mkl_convert(left_right), m, n, &a, lda,
                                         stride_a, &x, incx, stride_x, &c, ldc, stride_c,
                                         batch_size);
}

void gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, float alpha, sycl::buffer<float, 1> &a, int64_t lda,
                int64_t stride_a, sycl::buffer<float, 1> &b, int64_t ldb, int64_t stride_b,
                float beta, sycl::buffer<float, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    ::oneapi::mkl::gpu::sgemm_batch(queue, MAJOR, mkl_convert(transa), mkl_convert(transb), m, n, k,
                                    alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
                                    stride_c, batch_size);
}

void gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, double alpha, sycl::buffer<double, 1> &a, int64_t lda,
                int64_t stride_a, sycl::buffer<double, 1> &b, int64_t ldb, int64_t stride_b,
                double beta, sycl::buffer<double, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    ::oneapi::mkl::gpu::dgemm_batch(queue, MAJOR, mkl_convert(transa), mkl_convert(transb), m, n, k,
                                    alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
                                    stride_c, batch_size);
}

void gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a,
                int64_t lda, int64_t stride_a, sycl::buffer<std::complex<float>, 1> &b,
                int64_t ldb, int64_t stride_b, std::complex<float> beta,
                sycl::buffer<std::complex<float>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    ::oneapi::mkl::gpu::cgemm_batch(queue, MAJOR, mkl_convert(transa), mkl_convert(transb), m, n, k,
                                    alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
                                    stride_c, batch_size);
}

void gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
                int64_t lda, int64_t stride_a, sycl::buffer<std::complex<double>, 1> &b,
                int64_t ldb, int64_t stride_b, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    ::oneapi::mkl::gpu::zgemm_batch(queue, MAJOR, mkl_convert(transa), mkl_convert(transb), m, n, k,
                                    alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
                                    stride_c, batch_size);
}

void gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, sycl::half alpha, sycl::buffer<sycl::half, 1> &a, int64_t lda,
                int64_t stride_a, sycl::buffer<sycl::half, 1> &b, int64_t ldb, int64_t stride_b,
                sycl::half beta, sycl::buffer<sycl::half, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    ::oneapi::mkl::gpu::hgemm_batch(queue, MAJOR, mkl_convert(transa), mkl_convert(transb), m, n, k,
                                    alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
                                    stride_c, batch_size);
}

void trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, float alpha, sycl::buffer<float, 1> &a,
                int64_t lda, int64_t stride_a, sycl::buffer<float, 1> &b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
    ::oneapi::mkl::gpu::strsm_batch(queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower),
                                    mkl_convert(trans), mkl_convert(unit_diag), m, n, alpha, a, lda,
                                    stride_a, b, ldb, stride_b, batch_size);
}

void trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, double alpha, sycl::buffer<double, 1> &a,
                int64_t lda, int64_t stride_a, sycl::buffer<double, 1> &b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
    ::oneapi::mkl::gpu::dtrsm_batch(queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower),
                                    mkl_convert(trans), mkl_convert(unit_diag), m, n, alpha, a, lda,
                                    stride_a, b, ldb, stride_b, batch_size);
}

void trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
    ::oneapi::mkl::gpu::ctrsm_batch(queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower),
                                    mkl_convert(trans), mkl_convert(unit_diag), m, n, alpha, a, lda,
                                    stride_a, b, ldb, stride_b, batch_size);
}

void trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
    ::oneapi::mkl::gpu::ztrsm_batch(queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower),
                                    mkl_convert(trans), mkl_convert(unit_diag), m, n, alpha, a, lda,
                                    stride_a, b, ldb, stride_b, batch_size);
}

void syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                float alpha, sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a,
                float beta, sycl::buffer<float, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    ::oneapi::mkl::gpu::ssyrk_batch_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                         mkl_convert(trans), n, k, alpha, &a, lda, stride_a, beta,
                                         &c, ldc, stride_c, batch_size);
}

void syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                double alpha, sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                double beta, sycl::buffer<double, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    ::oneapi::mkl::gpu::dsyrk_batch_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                         mkl_convert(trans), n, k, alpha, &a, lda, stride_a, beta,
                                         &c, ldc, stride_c, batch_size);
}

void syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
                int64_t stride_a, std::complex<float> beta,
                sycl::buffer<std::complex<float>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    ::oneapi::mkl::gpu::csyrk_batch_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                         mkl_convert(trans), n, k, alpha, &a, lda, stride_a, beta,
                                         &c, ldc, stride_c, batch_size);
}

void syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
                int64_t lda, int64_t stride_a, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    ::oneapi::mkl::gpu::zsyrk_batch_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                         mkl_convert(trans), n, k, alpha, &a, lda, stride_a, beta,
                                         &c, ldc, stride_c, batch_size);
}

// USM APIs

sycl::event *coalesce_events(sycl::queue &queue, std::vector<sycl::event *> &prereqs) {
#ifdef _WIN64
    for (int64_t i = 0; i < prereqs.size(); i++)
        prereqs[i]->wait();
    return new sycl::event();
#else
    if (prereqs.size() > 0) {
        return new sycl::event(queue.submit([&](sycl::handler &cgh) {
            for (int64_t i = 0; i < prereqs.size(); i++)
                cgh.depends_on(*prereqs[i]);
            cgh.single_task<class EMPTY_KERNEL_NAME>([]() {});
        }));
    }
    else
        return new sycl::event();
#endif
}

sycl::event copy_batch(sycl::queue &queue, int64_t n, const float *x, int64_t incx,
                           std::int64_t stridex, float *y, int64_t incy, std::int64_t stridey,
                           std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::scopy_batch_sycl(&queue, n, x, incx, stridex, y, incy, stridey,
                                                batch_size, dependencies);
}

sycl::event copy_batch(sycl::queue &queue, int64_t n, const double *x, int64_t incx,
                           std::int64_t stridex, double *y, int64_t incy, std::int64_t stridey,
                           std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dcopy_batch_sycl(&queue, n, x, incx, stridex, y, incy, stridey,
                                                batch_size, dependencies);
}

sycl::event copy_batch(sycl::queue &queue, int64_t n, const std::complex<float> *x,
                           int64_t incx, std::int64_t stridex, std::complex<float> *y, int64_t incy,
                           std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ccopy_batch_sycl(&queue, n, x, incx, stridex, y, incy, stridey,
                                                batch_size, dependencies);
}

sycl::event copy_batch(sycl::queue &queue, int64_t n, const std::complex<double> *x,
                           int64_t incx, std::int64_t stridex, std::complex<double> *y,
                           int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zcopy_batch_sycl(&queue, n, x, incx, stridex, y, incy, stridey,
                                                batch_size, dependencies);
}

sycl::event copy_batch(sycl::queue &queue, int64_t *n, const float **x, int64_t *incx,
                           float **y, int64_t *incy, int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_group_size = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *copy_batch_event = new sycl::event(
            ::oneapi::mkl::gpu::scopy_batch_sycl(&queue, n[i], x, incx[i], y, incy[i],
                                                 group_size[i], total_group_size, dependencies));
        coalesced_events.push_back(copy_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event copy_batch(sycl::queue &queue, int64_t *n, const double **x, int64_t *incx,
                           double **y, int64_t *incy, int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_group_size = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *copy_batch_event = new sycl::event(
            ::oneapi::mkl::gpu::dcopy_batch_sycl(&queue, n[i], x, incx[i], y, incy[i],
                                                 group_size[i], total_group_size, dependencies));
        coalesced_events.push_back(copy_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event copy_batch(sycl::queue &queue, int64_t *n, const std::complex<float> **x,
                           int64_t *incx, std::complex<float> **y, int64_t *incy,
                           int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_group_size = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *copy_batch_event = new sycl::event(
            ::oneapi::mkl::gpu::ccopy_batch_sycl(&queue, n[i], x, incx[i], y, incy[i],
                                                 group_size[i], total_group_size, dependencies));
        coalesced_events.push_back(copy_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event copy_batch(sycl::queue &queue, int64_t *n, const std::complex<double> **x,
                           int64_t *incx, std::complex<double> **y, int64_t *incy,
                           int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_group_size = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *copy_batch_event = new sycl::event(
            ::oneapi::mkl::gpu::zcopy_batch_sycl(&queue, n[i], x, incx[i], y, incy[i],
                                                 group_size[i], total_group_size, dependencies));
        coalesced_events.push_back(copy_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event axpy_batch(sycl::queue &queue, int64_t n, float alpha, const float *x,
                           int64_t incx, int64_t stridex, float *y, int64_t incy, int64_t stridey,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::saxpy_batch_sycl(&queue, n, alpha, x, incx, stridex, y, incy,
                                                stridey, batch_size, dependencies);
}

sycl::event axpy_batch(sycl::queue &queue, int64_t n, double alpha, const double *x,
                           int64_t incx, int64_t stridex, double *y, int64_t incy, int64_t stridey,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::daxpy_batch_sycl(&queue, n, alpha, x, incx, stridex, y, incy,
                                                stridey, batch_size, dependencies);
}

sycl::event axpy_batch(sycl::queue &queue, int64_t n, std::complex<float> alpha,
                           const std::complex<float> *x, int64_t incx, int64_t stridex,
                           std::complex<float> *y, int64_t incy, int64_t stridey,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::caxpy_batch_sycl(&queue, n, alpha, x, incx, stridex, y, incy,
                                                stridey, batch_size, dependencies);
}

sycl::event axpy_batch(sycl::queue &queue, int64_t n, std::complex<double> alpha,
                           const std::complex<double> *x, int64_t incx, int64_t stridex,
                           std::complex<double> *y, int64_t incy, int64_t stridey,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zaxpy_batch_sycl(&queue, n, alpha, x, incx, stridex, y, incy,
                                                stridey, batch_size, dependencies);
}

sycl::event axpy_batch(sycl::queue &queue, int64_t *n, float *alpha, const float **x,
                           int64_t *incx, float **y, int64_t *incy, int64_t group_count,
                           int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_group_size = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *axpy_batch_event = new sycl::event(
            ::oneapi::mkl::gpu::saxpy_batch_sycl(&queue, n[i], alpha[i], x, incx[i], y, incy[i],
                                                 group_size[i], total_group_size, dependencies));
        coalesced_events.push_back(axpy_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event axpy_batch(sycl::queue &queue, int64_t *n, double *alpha, const double **x,
                           int64_t *incx, double **y, int64_t *incy, int64_t group_count,
                           int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_group_size = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *axpy_batch_event = new sycl::event(
            ::oneapi::mkl::gpu::daxpy_batch_sycl(&queue, n[i], alpha[i], x, incx[i], y, incy[i],
                                                 group_size[i], total_group_size, dependencies));
        coalesced_events.push_back(axpy_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event axpy_batch(sycl::queue &queue, int64_t *n, std::complex<float> *alpha,
                           const std::complex<float> **x, int64_t *incx, std::complex<float> **y,
                           int64_t *incy, int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_group_size = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *axpy_batch_event = new sycl::event(
            ::oneapi::mkl::gpu::caxpy_batch_sycl(&queue, n[i], alpha[i], x, incx[i], y, incy[i],
                                                 group_size[i], total_group_size, dependencies));
        coalesced_events.push_back(axpy_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event axpy_batch(sycl::queue &queue, int64_t *n, std::complex<double> *alpha,
                           const std::complex<double> **x, int64_t *incx, std::complex<double> **y,
                           int64_t *incy, int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_group_size = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *axpy_batch_event = new sycl::event(
            ::oneapi::mkl::gpu::zaxpy_batch_sycl(&queue, n[i], alpha[i], x, incx[i], y, incy[i],
                                                 group_size[i], total_group_size, dependencies));
        coalesced_events.push_back(axpy_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                           float alpha, const float *a, int64_t lda, int64_t stride_a,
                           const float *x, int64_t incx, int64_t stride_x, float beta, float *y,
                           int64_t incy, int64_t stride_y, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sgemv_batch_sycl(&queue, MAJOR, mkl_convert(transa), m, n, alpha, a,
                                                lda, stride_a, x, incx, stride_x, beta, y, incy,
                                                stride_y, batch_size, dependencies);
}

sycl::event gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                           double alpha, const double *a, int64_t lda, int64_t stride_a,
                           const double *x, int64_t incx, int64_t stride_x, double beta, double *y,
                           int64_t incy, int64_t stride_y, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dgemv_batch_sycl(&queue, MAJOR, mkl_convert(transa), m, n, alpha, a,
                                                lda, stride_a, x, incx, stride_x, beta, y, incy,
                                                stride_y, batch_size, dependencies);
}

sycl::event gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                           std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                           int64_t stride_a, const std::complex<float> *x, int64_t incx,
                           int64_t stride_x, std::complex<float> beta, std::complex<float> *y,
                           int64_t incy, int64_t stride_y, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgemv_batch_sycl(&queue, MAJOR, mkl_convert(transa), m, n, alpha, a,
                                                lda, stride_a, x, incx, stride_x, beta, y, incy,
                                                stride_y, batch_size, dependencies);
}

sycl::event gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                           std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                           int64_t stride_a, const std::complex<double> *x, int64_t incx,
                           int64_t stride_x, std::complex<double> beta, std::complex<double> *y,
                           int64_t incy, int64_t stride_y, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgemv_batch_sycl(&queue, MAJOR, mkl_convert(transa), m, n, alpha, a,
                                                lda, stride_a, x, incx, stride_x, beta, y, incy,
                                                stride_y, batch_size, dependencies);
}

sycl::event gemv_batch(sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n,
                           float *alpha, const float **a, int64_t *lda, const float **x,
                           int64_t *incx, float *beta, float **y, int64_t *incy,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_groupsize = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *gemv_batch_event =
            new sycl::event(::oneapi::mkl::gpu::sgemv_batch_sycl(
                &queue, MAJOR, mkl_convert(transa[i]), m[i], n[i], alpha[i], a, lda[i], x, incx[i],
                beta[i], y, incy[i], total_groupsize, groupsize[i], dependencies));
        coalesced_events.push_back(gemv_batch_event);
        total_groupsize += groupsize[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event gemv_batch(sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n,
                           double *alpha, const double **a, int64_t *lda, const double **x,
                           int64_t *incx, double *beta, double **y, int64_t *incy,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_groupsize = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *gemv_batch_event =
            new sycl::event(::oneapi::mkl::gpu::dgemv_batch_sycl(
                &queue, MAJOR, mkl_convert(transa[i]), m[i], n[i], alpha[i], a, lda[i], x, incx[i],
                beta[i], y, incy[i], total_groupsize, groupsize[i], dependencies));
        coalesced_events.push_back(gemv_batch_event);
        total_groupsize += groupsize[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event gemv_batch(sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n,
                           std::complex<float> *alpha, const std::complex<float> **a, int64_t *lda,
                           const std::complex<float> **x, int64_t *incx, std::complex<float> *beta,
                           std::complex<float> **y, int64_t *incy, int64_t group_count,
                           int64_t *groupsize, const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_groupsize = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *gemv_batch_event =
            new sycl::event(::oneapi::mkl::gpu::cgemv_batch_sycl(
                &queue, MAJOR, mkl_convert(transa[i]), m[i], n[i], alpha[i], a, lda[i], x, incx[i],
                beta[i], y, incy[i], total_groupsize, groupsize[i], dependencies));
        coalesced_events.push_back(gemv_batch_event);
        total_groupsize += groupsize[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event gemv_batch(sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n,
                           std::complex<double> *alpha, const std::complex<double> **a,
                           int64_t *lda, const std::complex<double> **x, int64_t *incx,
                           std::complex<double> *beta, std::complex<double> **y, int64_t *incy,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_groupsize = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *gemv_batch_event =
            new sycl::event(::oneapi::mkl::gpu::zgemv_batch_sycl(
                &queue, MAJOR, mkl_convert(transa[i]), m[i], n[i], alpha[i], a, lda[i], x, incx[i],
                beta[i], y, incy[i], total_groupsize, groupsize[i], dependencies));
        coalesced_events.push_back(gemv_batch_event);
        total_groupsize += groupsize[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                           const float *a, int64_t lda, int64_t stride_a, const float *x,
                           int64_t incx, int64_t stride_x, float *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sdgmm_batch_sycl(&queue, MAJOR, mkl_convert(left_right), m, n, a,
                                                lda, stride_a, x, incx, stride_x, c, ldc, stride_c,
                                                batch_size, dependencies);
}

sycl::event dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                           const double *a, int64_t lda, int64_t stride_a, const double *x,
                           int64_t incx, int64_t stride_x, double *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ddgmm_batch_sycl(&queue, MAJOR, mkl_convert(left_right), m, n, a,
                                                lda, stride_a, x, incx, stride_x, c, ldc, stride_c,
                                                batch_size, dependencies);
}

sycl::event dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                           const std::complex<float> *a, int64_t lda, int64_t stride_a,
                           const std::complex<float> *x, int64_t incx, int64_t stride_x,
                           std::complex<float> *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cdgmm_batch_sycl(&queue, MAJOR, mkl_convert(left_right), m, n, a,
                                                lda, stride_a, x, incx, stride_x, c, ldc, stride_c,
                                                batch_size, dependencies);
}

sycl::event dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                           const std::complex<double> *a, int64_t lda, int64_t stride_a,
                           const std::complex<double> *x, int64_t incx, int64_t stride_x,
                           std::complex<double> *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zdgmm_batch_sycl(&queue, MAJOR, mkl_convert(left_right), m, n, a,
                                                lda, stride_a, x, incx, stride_x, c, ldc, stride_c,
                                                batch_size, dependencies);
}

sycl::event dgmm_batch(sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                           const float **a, int64_t *lda, const float **x, int64_t *incx, float **c,
                           int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_groupsize = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *dgmm_batch_event =
            new sycl::event(::oneapi::mkl::gpu::sdgmm_batch_sycl(
                &queue, MAJOR, mkl_convert(left_right[i]), m[i], n[i], a, lda[i], x, incx[i], c,
                ldc[i], total_groupsize, groupsize[i], dependencies));
        coalesced_events.push_back(dgmm_batch_event);
        total_groupsize += groupsize[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event dgmm_batch(sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                           const double **a, int64_t *lda, const double **x, int64_t *incx,
                           double **c, int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_groupsize = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *dgmm_batch_event =
            new sycl::event(::oneapi::mkl::gpu::ddgmm_batch_sycl(
                &queue, MAJOR, mkl_convert(left_right[i]), m[i], n[i], a, lda[i], x, incx[i], c,
                ldc[i], total_groupsize, groupsize[i], dependencies));
        coalesced_events.push_back(dgmm_batch_event);
        total_groupsize += groupsize[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event dgmm_batch(sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                           const std::complex<float> **a, int64_t *lda,
                           const std::complex<float> **x, int64_t *incx, std::complex<float> **c,
                           int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_groupsize = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *dgmm_batch_event =
            new sycl::event(::oneapi::mkl::gpu::cdgmm_batch_sycl(
                &queue, MAJOR, mkl_convert(left_right[i]), m[i], n[i], a, lda[i], x, incx[i], c,
                ldc[i], total_groupsize, groupsize[i], dependencies));
        coalesced_events.push_back(dgmm_batch_event);
        total_groupsize += groupsize[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event dgmm_batch(sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                           const std::complex<double> **a, int64_t *lda,
                           const std::complex<double> **x, int64_t *incx, std::complex<double> **c,
                           int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_groupsize = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *dgmm_batch_event =
            new sycl::event(::oneapi::mkl::gpu::zdgmm_batch_sycl(
                &queue, MAJOR, mkl_convert(left_right[i]), m[i], n[i], a, lda[i], x, incx[i], c,
                ldc[i], total_groupsize, groupsize[i], dependencies));
        coalesced_events.push_back(dgmm_batch_event);
        total_groupsize += groupsize[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                           int64_t n, int64_t k, float alpha, const float *a, int64_t lda,
                           int64_t stride_a, const float *b, int64_t ldb, int64_t stride_b,
                           float beta, float *c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sgemm_batch_sycl(
        &queue, MAJOR, mkl_convert(transa), mkl_convert(transb), m, n, k, alpha, a, lda, stride_a,
        b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
}

sycl::event gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                           int64_t n, int64_t k, double alpha, const double *a, int64_t lda,
                           int64_t stride_a, const double *b, int64_t ldb, int64_t stride_b,
                           double beta, double *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dgemm_batch_sycl(
        &queue, MAJOR, mkl_convert(transa), mkl_convert(transb), m, n, k, alpha, a, lda, stride_a,
        b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
}

sycl::event gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                           int64_t n, int64_t k, std::complex<float> alpha,
                           const std::complex<float> *a, int64_t lda, int64_t stride_a,
                           const std::complex<float> *b, int64_t ldb, int64_t stride_b,
                           std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                           int64_t stride_c, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgemm_batch_sycl(
        &queue, MAJOR, mkl_convert(transa), mkl_convert(transb), m, n, k, alpha, a, lda, stride_a,
        b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
}

sycl::event gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                           int64_t n, int64_t k, std::complex<double> alpha,
                           const std::complex<double> *a, int64_t lda, int64_t stride_a,
                           const std::complex<double> *b, int64_t ldb, int64_t stride_b,
                           std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                           int64_t stride_c, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgemm_batch_sycl(
        &queue, MAJOR, mkl_convert(transa), mkl_convert(transb), m, n, k, alpha, a, lda, stride_a,
        b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
}

sycl::event gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                           int64_t n, int64_t k, sycl::half alpha, const sycl::half *a, int64_t lda,
                           int64_t stride_a, const sycl::half *b, int64_t ldb, int64_t stride_b,
                           sycl::half beta, sycl::half *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::hgemm_batch_sycl(
        &queue, MAJOR, mkl_convert(transa), mkl_convert(transb), m, n, k, alpha, a, lda, stride_a,
        b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
}

sycl::event gemm_batch(sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,
                           int64_t *n, int64_t *k, float *alpha, const float **a, int64_t *lda,
                           const float **b, int64_t *ldb, float *beta, float **c, int64_t *ldc,
                           int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_group_size = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *gemm_batch_event =
            new sycl::event(::oneapi::mkl::gpu::sgemm_batch_sycl(
                &queue, MAJOR, mkl_convert(transa[i]), mkl_convert(transb[i]), m[i], n[i], k[i],
                alpha[i], a, lda[i], b, ldb[i], beta[i], c, ldc[i], total_group_size, group_size[i],
                dependencies));
        coalesced_events.push_back(gemm_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event gemm_batch(sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,
                           int64_t *n, int64_t *k, double *alpha, const double **a, int64_t *lda,
                           const double **b, int64_t *ldb, double *beta, double **c, int64_t *ldc,
                           int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_group_size = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *gemm_batch_event =
            new sycl::event(::oneapi::mkl::gpu::dgemm_batch_sycl(
                &queue, MAJOR, mkl_convert(transa[i]), mkl_convert(transb[i]), m[i], n[i], k[i],
                alpha[i], a, lda[i], b, ldb[i], beta[i], c, ldc[i], total_group_size, group_size[i],
                dependencies));
        coalesced_events.push_back(gemm_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event gemm_batch(sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,
                           int64_t *n, int64_t *k, std::complex<float> *alpha,
                           const std::complex<float> **a, int64_t *lda,
                           const std::complex<float> **b, int64_t *ldb, std::complex<float> *beta,
                           std::complex<float> **c, int64_t *ldc, int64_t group_count,
                           int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_group_size = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *gemm_batch_event =
            new sycl::event(::oneapi::mkl::gpu::cgemm_batch_sycl(
                &queue, MAJOR, mkl_convert(transa[i]), mkl_convert(transb[i]), m[i], n[i], k[i],
                alpha[i], a, lda[i], b, ldb[i], beta[i], c, ldc[i], total_group_size, group_size[i],
                dependencies));
        coalesced_events.push_back(gemm_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event gemm_batch(sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,
                           int64_t *n, int64_t *k, std::complex<double> *alpha,
                           const std::complex<double> **a, int64_t *lda,
                           const std::complex<double> **b, int64_t *ldb, std::complex<double> *beta,
                           std::complex<double> **c, int64_t *ldc, int64_t group_count,
                           int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_group_size = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *gemm_batch_event =
            new sycl::event(::oneapi::mkl::gpu::zgemm_batch_sycl(
                &queue, MAJOR, mkl_convert(transa[i]), mkl_convert(transb[i]), m[i], n[i], k[i],
                alpha[i], a, lda[i], b, ldb[i], beta[i], c, ldc[i], total_group_size, group_size[i],
                dependencies));
        coalesced_events.push_back(gemm_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event gemm_batch(sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,
                           int64_t *n, int64_t *k, sycl::half *alpha, const sycl::half **a,
                           int64_t *lda, const sycl::half **b, int64_t *ldb, sycl::half *beta,
                           sycl::half **c, int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_groupsize = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *gemm_batch_event =
            new sycl::event(::oneapi::mkl::gpu::hgemm_batch_sycl(
                &queue, MAJOR, mkl_convert(transa[i]), mkl_convert(transb[i]), m[i], n[i], k[i],
                alpha[i], a, lda[i], b, ldb[i], beta[i], c, ldc[i], total_groupsize, groupsize[i],
                dependencies));
        coalesced_events.push_back(gemm_batch_event);
        total_groupsize += groupsize[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, int64_t m, int64_t n, float alpha,
                           const float *a, int64_t lda, int64_t stride_a, float *b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::strsm_batch_sycl(
        &queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower), mkl_convert(trans),
        mkl_convert(unit_diag), m, n, alpha, a, lda, stride_a, b, ldb, stride_b, batch_size,
        dependencies);
}

sycl::event trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, int64_t m, int64_t n, double alpha,
                           const double *a, int64_t lda, int64_t stride_a, double *b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtrsm_batch_sycl(
        &queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower), mkl_convert(trans),
        mkl_convert(unit_diag), m, n, alpha, a, lda, stride_a, b, ldb, stride_b, batch_size,
        dependencies);
}

sycl::event trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, int64_t m, int64_t n,
                           std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                           int64_t stride_a, std::complex<float> *b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctrsm_batch_sycl(
        &queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower), mkl_convert(trans),
        mkl_convert(unit_diag), m, n, alpha, a, lda, stride_a, b, ldb, stride_b, batch_size,
        dependencies);
}

sycl::event trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, int64_t m, int64_t n,
                           std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                           int64_t stride_a, std::complex<double> *b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztrsm_batch_sycl(
        &queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower), mkl_convert(trans),
        mkl_convert(unit_diag), m, n, alpha, a, lda, stride_a, b, ldb, stride_b, batch_size,
        dependencies);
}

sycl::event trsm_batch(sycl::queue &queue, side *left_right, uplo *upper_lower,
                           transpose *trans, diag *unit_diag, int64_t *m, int64_t *n, float *alpha,
                           const float **a, int64_t *lda, float **b, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_groupsize = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *trsm_batch_event =
            new sycl::event(::oneapi::mkl::gpu::strsm_batch_sycl(
                &queue, MAJOR, mkl_convert(left_right[i]), mkl_convert(upper_lower[i]),
                mkl_convert(trans[i]), mkl_convert(unit_diag[i]), m[i], n[i], alpha[i], a, lda[i],
                b, ldb[i], total_groupsize, groupsize[i], dependencies));
        coalesced_events.push_back(trsm_batch_event);
        total_groupsize += groupsize[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event trsm_batch(sycl::queue &queue, side *left_right, uplo *upper_lower,
                           transpose *trans, diag *unit_diag, int64_t *m, int64_t *n, double *alpha,
                           const double **a, int64_t *lda, double **b, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_groupsize = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *trsm_batch_event =
            new sycl::event(::oneapi::mkl::gpu::dtrsm_batch_sycl(
                &queue, MAJOR, mkl_convert(left_right[i]), mkl_convert(upper_lower[i]),
                mkl_convert(trans[i]), mkl_convert(unit_diag[i]), m[i], n[i], alpha[i], a, lda[i],
                b, ldb[i], total_groupsize, groupsize[i], dependencies));
        coalesced_events.push_back(trsm_batch_event);
        total_groupsize += groupsize[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event trsm_batch(sycl::queue &queue, side *left_right, uplo *upper_lower,
                           transpose *trans, diag *unit_diag, int64_t *m, int64_t *n,
                           std::complex<float> *alpha, const std::complex<float> **a, int64_t *lda,
                           std::complex<float> **b, int64_t *ldb, int64_t group_count,
                           int64_t *groupsize, const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_groupsize = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *trsm_batch_event =
            new sycl::event(::oneapi::mkl::gpu::ctrsm_batch_sycl(
                &queue, MAJOR, mkl_convert(left_right[i]), mkl_convert(upper_lower[i]),
                mkl_convert(trans[i]), mkl_convert(unit_diag[i]), m[i], n[i], alpha[i], a, lda[i],
                b, ldb[i], total_groupsize, groupsize[i], dependencies));
        coalesced_events.push_back(trsm_batch_event);
        total_groupsize += groupsize[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event trsm_batch(sycl::queue &queue, side *left_right, uplo *upper_lower,
                           transpose *trans, diag *unit_diag, int64_t *m, int64_t *n,
                           std::complex<double> *alpha, const std::complex<double> **a,
                           int64_t *lda, std::complex<double> **b, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_groupsize = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *trsm_batch_event =
            new sycl::event(::oneapi::mkl::gpu::ztrsm_batch_sycl(
                &queue, MAJOR, mkl_convert(left_right[i]), mkl_convert(upper_lower[i]),
                mkl_convert(trans[i]), mkl_convert(unit_diag[i]), m[i], n[i], alpha[i], a, lda[i],
                b, ldb[i], total_groupsize, groupsize[i], dependencies));
        coalesced_events.push_back(trsm_batch_event);
        total_groupsize += groupsize[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                           int64_t k, float alpha, const float *a, int64_t lda, int64_t stride_a,
                           float beta, float *c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssyrk_batch_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                                mkl_convert(trans), n, k, alpha, a, lda, stride_a,
                                                beta, c, ldc, stride_c, batch_size, dependencies);
}

sycl::event syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                           int64_t k, double alpha, const double *a, int64_t lda, int64_t stride_a,
                           double beta, double *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsyrk_batch_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                                mkl_convert(trans), n, k, alpha, a, lda, stride_a,
                                                beta, c, ldc, stride_c, batch_size, dependencies);
}

sycl::event syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                           int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                           int64_t lda, int64_t stride_a, std::complex<float> beta,
                           std::complex<float> *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::csyrk_batch_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                                mkl_convert(trans), n, k, alpha, a, lda, stride_a,
                                                beta, c, ldc, stride_c, batch_size, dependencies);
}

sycl::event syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                           int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                           int64_t lda, int64_t stride_a, std::complex<double> beta,
                           std::complex<double> *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zsyrk_batch_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                                mkl_convert(trans), n, k, alpha, a, lda, stride_a,
                                                beta, c, ldc, stride_c, batch_size, dependencies);
}

sycl::event syrk_batch(sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                           int64_t *k, float *alpha, const float **a, int64_t *lda, float *beta,
                           float **c, int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_groupsize = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *syrk_batch_event =
            new sycl::event(::oneapi::mkl::gpu::ssyrk_batch_sycl(
                &queue, MAJOR, mkl_convert(upper_lower[i]), mkl_convert(trans[i]), n[i], k[i],
                alpha[i], a, lda[i], beta[i], c, ldc[i], total_groupsize, groupsize[i],
                dependencies));
        coalesced_events.push_back(syrk_batch_event);
        total_groupsize += groupsize[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event syrk_batch(sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                           int64_t *k, double *alpha, const double **a, int64_t *lda, double *beta,
                           double **c, int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_groupsize = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *syrk_batch_event =
            new sycl::event(::oneapi::mkl::gpu::dsyrk_batch_sycl(
                &queue, MAJOR, mkl_convert(upper_lower[i]), mkl_convert(trans[i]), n[i], k[i],
                alpha[i], a, lda[i], beta[i], c, ldc[i], total_groupsize, groupsize[i],
                dependencies));
        coalesced_events.push_back(syrk_batch_event);
        total_groupsize += groupsize[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event syrk_batch(sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                           int64_t *k, std::complex<float> *alpha, const std::complex<float> **a,
                           int64_t *lda, std::complex<float> *beta, std::complex<float> **c,
                           int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_groupsize = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *syrk_batch_event =
            new sycl::event(::oneapi::mkl::gpu::csyrk_batch_sycl(
                &queue, MAJOR, mkl_convert(upper_lower[i]), mkl_convert(trans[i]), n[i], k[i],
                alpha[i], a, lda[i], beta[i], c, ldc[i], total_groupsize, groupsize[i],
                dependencies));
        coalesced_events.push_back(syrk_batch_event);
        total_groupsize += groupsize[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

sycl::event syrk_batch(sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                           int64_t *k, std::complex<double> *alpha, const std::complex<double> **a,
                           int64_t *lda, std::complex<double> *beta, std::complex<double> **c,
                           int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    std::vector<sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    int64_t total_groupsize = 0;
    for (int64_t i = 0; i < group_count; i++) {
        sycl::event *syrk_batch_event =
            new sycl::event(::oneapi::mkl::gpu::zsyrk_batch_sycl(
                &queue, MAJOR, mkl_convert(upper_lower[i]), mkl_convert(trans[i]), n[i], k[i],
                alpha[i], a, lda[i], beta[i], c, ldc[i], total_groupsize, groupsize[i],
                dependencies));
        coalesced_events.push_back(syrk_batch_event);
        total_groupsize += groupsize[i];
    }
    return *coalesce_events(queue, coalesced_events);
}
