/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/
#include "cublas_helper.hpp"
#include "cublas_task.hpp"
#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/blas/detail/cublas/onemkl_blas_cublas.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace cublas {
namespace column_major {

// Buffer APIs

void copy_batch(sycl::queue &queue, int64_t n, sycl::buffer<float, 1> &x, int64_t incx,
                int64_t stridex, sycl::buffer<float, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

void copy_batch(sycl::queue &queue, int64_t n, sycl::buffer<double, 1> &x, int64_t incx,
                int64_t stridex, sycl::buffer<double, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

void copy_batch(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<float>, 1> &x,
                int64_t incx, int64_t stridex, sycl::buffer<std::complex<float>, 1> &y,
                int64_t incy, int64_t stridey, int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

void copy_batch(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<double>, 1> &x,
                int64_t incx, int64_t stridex, sycl::buffer<std::complex<double>, 1> &y,
                int64_t incy, int64_t stridey, int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

void axpy_batch(sycl::queue &queue, int64_t n, float alpha, sycl::buffer<float, 1> &x, int64_t incx,
                int64_t stridex, sycl::buffer<float, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

void axpy_batch(sycl::queue &queue, int64_t n, double alpha, sycl::buffer<double, 1> &x,
                int64_t incx, int64_t stridex, sycl::buffer<double, 1> &y, int64_t incy,
                int64_t stridey, int64_t batch_size) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

void axpy_batch(sycl::queue &queue, int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1> &x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<float>, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

void axpy_batch(sycl::queue &queue, int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1> &x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<double>, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

void gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n, float alpha,
                sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a, sycl::buffer<float, 1> &x,
                int64_t incx, int64_t stride_x, float beta, sycl::buffer<float, 1> &y, int64_t incy,
                int64_t stride_y, int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

void gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n, double alpha,
                sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<double, 1> &x, int64_t incx, int64_t stride_x, double beta,
                sycl::buffer<double, 1> &y, int64_t incy, int64_t stride_y, int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

void gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
                int64_t stride_x, std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &y,
                int64_t incy, int64_t stride_y, int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

void gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
                int64_t stride_x, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1> &y, int64_t incy, int64_t stride_y,
                int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

void dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a, sycl::buffer<float, 1> &x,
                int64_t incx, int64_t stride_x, sycl::buffer<float, 1> &c, int64_t ldc,
                int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

void dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<double, 1> &x, int64_t incx, int64_t stride_x,
                sycl::buffer<double, 1> &c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

void dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<std::complex<float>, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<float>, 1> &x, int64_t incx, int64_t stride_x,
                sycl::buffer<std::complex<float>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

void dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<std::complex<double>, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<double>, 1> &x, int64_t incx, int64_t stride_x,
                sycl::buffer<std::complex<double>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

template <typename Func, typename T>
inline void gemm_batch(const char *func_name, Func func, sycl::queue &queue, transpose transa,
                       transpose transb, int64_t m, int64_t n, int64_t k, T alpha,
                       sycl::buffer<T, 1> &a, int64_t lda, int64_t stride_a, sycl::buffer<T, 1> &b,
                       int64_t ldb, int64_t stride_b, T beta, sycl::buffer<T, 1> &c, int64_t ldc,
                       int64_t stride_c, int64_t batch_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_size);
    queue.submit([&](sycl::handler &cgh) {
        if (!verify_support<sycl::half, T>(queue, sycl::aspect::fp16)) {
            throw oneapi::mkl::unimplemented(
                "blas", "sycl::half", "half is not supported by the device or the sycl compiler");
        }
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType *>(a_acc);
            auto b_ = sc.get_mem<cuDataType *>(b_acc);
            auto c_ = sc.get_mem<cuDataType *>(c_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_operation(transa),
                                     get_cublas_operation(transb), m, n, k, (cuDataType *)&alpha,
                                     a_, lda, stride_a, b_, ldb, stride_b, (cuDataType *)&beta, c_,
                                     ldc, stride_c, batch_size);
        });
    });
}

#define GEMM_STRIDED_BATCH_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                          \
    void gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,  \
                    int64_t k, TYPE alpha, sycl::buffer<TYPE, 1> &a, int64_t lda,                  \
                    int64_t stride_a, sycl::buffer<TYPE, 1> &b, int64_t ldb, int64_t stride_b,     \
                    TYPE beta, sycl::buffer<TYPE, 1> &c, int64_t ldc, int64_t stride_c,            \
                    int64_t batch_size) {                                                          \
        gemm_batch(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, \
                   stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);                \
    }

GEMM_STRIDED_BATCH_LAUNCHER(sycl::half, cublasHgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER(float, cublasSgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER(double, cublasDgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER(std::complex<float>, cublasCgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER(std::complex<double>, cublasZgemmStridedBatched)

#undef GEMM_STRIDED_BATCH_LAUNCHER

void trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, float alpha, sycl::buffer<float, 1> &a,
                int64_t lda, int64_t stride_a, sycl::buffer<float, 1> &b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

void trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, double alpha, sycl::buffer<double, 1> &a,
                int64_t lda, int64_t stride_a, sycl::buffer<double, 1> &b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

void trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

void trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

void syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                float alpha, sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a, float beta,
                sycl::buffer<float, 1> &c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

void syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                double alpha, sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                double beta, sycl::buffer<double, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

void syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
                int64_t stride_a, std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &c,
                int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

void syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
                int64_t stride_a, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

void omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                    sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a,
                    sycl::buffer<float, 1> &b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

void omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                    sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                    sycl::buffer<double, 1> &b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

void omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
                    int64_t stride_a, sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
                    int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

void omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
                    int64_t lda, int64_t stride_a, sycl::buffer<std::complex<double>, 1> &b,
                    int64_t ldb, int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                    sycl::buffer<float, 1> &ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                    sycl::buffer<double, 1> &ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

void omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   float alpha, sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a,
                   float beta, sycl::buffer<float, 1> &b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<float, 1> &c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
}

void omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   double alpha, sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                   double beta, sycl::buffer<double, 1> &b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<double, 1> &c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
}

void omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
                   int64_t stride_a, std::complex<float> beta,
                   sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<std::complex<float>, 1> &c, int64_t ldc, int64_t stride_c,
                   int64_t batch_size) {
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
}

void omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
                   int64_t lda, int64_t stride_a, std::complex<double> beta,
                   sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<std::complex<double>, 1> &c, int64_t ldc, int64_t stride_c,
                   int64_t batch_size) {
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
}

// USM APIs

sycl::event copy_batch(sycl::queue &queue, int64_t *n, const float **x, int64_t *incx, float **y,
                       int64_t *incy, int64_t group_count, int64_t *group_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

sycl::event copy_batch(sycl::queue &queue, int64_t *n, const double **x, int64_t *incx, double **y,
                       int64_t *incy, int64_t group_count, int64_t *group_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

sycl::event copy_batch(sycl::queue &queue, int64_t *n, const std::complex<float> **x, int64_t *incx,
                       std::complex<float> **y, int64_t *incy, int64_t group_count,
                       int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

sycl::event copy_batch(sycl::queue &queue, int64_t *n, const std::complex<double> **x,
                       int64_t *incx, std::complex<double> **y, int64_t *incy, int64_t group_count,
                       int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

sycl::event copy_batch(sycl::queue &queue, int64_t n, const float *x, int64_t incx,
                       std::int64_t stridex, float *y, int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

sycl::event copy_batch(sycl::queue &queue, int64_t n, const double *x, int64_t incx,
                       std::int64_t stridex, double *y, int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

sycl::event copy_batch(sycl::queue &queue, int64_t n, const std::complex<float> *x, int64_t incx,
                       std::int64_t stridex, std::complex<float> *y, int64_t incy,
                       std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

sycl::event copy_batch(sycl::queue &queue, int64_t n, const std::complex<double> *x, int64_t incx,
                       std::int64_t stridex, std::complex<double> *y, int64_t incy,
                       std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

sycl::event axpy_batch(sycl::queue &queue, int64_t *n, float *alpha, const float **x, int64_t *incx,
                       float **y, int64_t *incy, int64_t group_count, int64_t *group_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

sycl::event axpy_batch(sycl::queue &queue, int64_t *n, double *alpha, const double **x,
                       int64_t *incx, double **y, int64_t *incy, int64_t group_count,
                       int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

sycl::event axpy_batch(sycl::queue &queue, int64_t *n, std::complex<float> *alpha,
                       const std::complex<float> **x, int64_t *incx, std::complex<float> **y,
                       int64_t *incy, int64_t group_count, int64_t *group_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

sycl::event axpy_batch(sycl::queue &queue, int64_t *n, std::complex<double> *alpha,
                       const std::complex<double> **x, int64_t *incx, std::complex<double> **y,
                       int64_t *incy, int64_t group_count, int64_t *group_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

sycl::event axpy_batch(sycl::queue &queue, int64_t n, float alpha, const float *x, int64_t incx,
                       int64_t stridex, float *y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

sycl::event axpy_batch(sycl::queue &queue, int64_t n, double alpha, const double *x, int64_t incx,
                       int64_t stridex, double *y, int64_t incy, int64_t stridey,
                       int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

sycl::event axpy_batch(sycl::queue &queue, int64_t n, std::complex<float> alpha,
                       const std::complex<float> *x, int64_t incx, int64_t stridex,
                       std::complex<float> *y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

sycl::event axpy_batch(sycl::queue &queue, int64_t n, std::complex<double> alpha,
                       const std::complex<double> *x, int64_t incx, int64_t stridex,
                       std::complex<double> *y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

sycl::event gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n, float alpha,
                       const float *a, int64_t lda, int64_t stride_a, const float *x, int64_t incx,
                       int64_t stride_x, float beta, float *y, int64_t incy, int64_t stride_y,
                       int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

sycl::event gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n, double alpha,
                       const double *a, int64_t lda, int64_t stride_a, const double *x,
                       int64_t incx, int64_t stride_x, double beta, double *y, int64_t incy,
                       int64_t stride_y, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

sycl::event gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                       std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                       int64_t stride_a, const std::complex<float> *x, int64_t incx,
                       int64_t stride_x, std::complex<float> beta, std::complex<float> *y,
                       int64_t incy, int64_t stride_y, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

sycl::event gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                       std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                       int64_t stride_a, const std::complex<double> *x, int64_t incx,
                       int64_t stride_x, std::complex<double> beta, std::complex<double> *y,
                       int64_t incy, int64_t stride_y, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

sycl::event gemv_batch(sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n, float *alpha,
                       const float **a, int64_t *lda, const float **x, int64_t *incx, float *beta,
                       float **y, int64_t *incy, int64_t group_count, int64_t *groupsize,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

sycl::event gemv_batch(sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n, double *alpha,
                       const double **a, int64_t *lda, const double **x, int64_t *incx,
                       double *beta, double **y, int64_t *incy, int64_t group_count,
                       int64_t *groupsize, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

sycl::event gemv_batch(sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n,
                       std::complex<float> *alpha, const std::complex<float> **a, int64_t *lda,
                       const std::complex<float> **x, int64_t *incx, std::complex<float> *beta,
                       std::complex<float> **y, int64_t *incy, int64_t group_count,
                       int64_t *groupsize, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

sycl::event gemv_batch(sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n,
                       std::complex<double> *alpha, const std::complex<double> **a, int64_t *lda,
                       const std::complex<double> **x, int64_t *incx, std::complex<double> *beta,
                       std::complex<double> **y, int64_t *incy, int64_t group_count,
                       int64_t *groupsize, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

sycl::event dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n, const float *a,
                       int64_t lda, int64_t stride_a, const float *x, int64_t incx,
                       int64_t stride_x, float *c, int64_t ldc, int64_t stride_c,
                       int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

sycl::event dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n, const double *a,
                       int64_t lda, int64_t stride_a, const double *x, int64_t incx,
                       int64_t stride_x, double *c, int64_t ldc, int64_t stride_c,
                       int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

sycl::event dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                       const std::complex<float> *a, int64_t lda, int64_t stride_a,
                       const std::complex<float> *x, int64_t incx, int64_t stride_x,
                       std::complex<float> *c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

sycl::event dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                       const std::complex<double> *a, int64_t lda, int64_t stride_a,
                       const std::complex<double> *x, int64_t incx, int64_t stride_x,
                       std::complex<double> *c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

sycl::event dgmm_batch(sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                       const float **a, int64_t *lda, const float **x, int64_t *incx, float **c,
                       int64_t *ldc, int64_t group_count, int64_t *groupsize,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

sycl::event dgmm_batch(sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                       const double **a, int64_t *lda, const double **x, int64_t *incx, double **c,
                       int64_t *ldc, int64_t group_count, int64_t *groupsize,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

sycl::event dgmm_batch(sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                       const std::complex<float> **a, int64_t *lda, const std::complex<float> **x,
                       int64_t *incx, std::complex<float> **c, int64_t *ldc, int64_t group_count,
                       int64_t *groupsize, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

sycl::event dgmm_batch(sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                       const std::complex<double> **a, int64_t *lda, const std::complex<double> **x,
                       int64_t *incx, std::complex<double> **c, int64_t *ldc, int64_t group_count,
                       int64_t *groupsize, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

template <typename Func, typename T>
inline sycl::event gemm_batch(const char *func_name, Func func, sycl::queue &queue,
                              transpose transa, transpose transb, int64_t m, int64_t n, int64_t k,
                              T alpha, const T *a, int64_t lda, int64_t stride_a, const T *b,
                              int64_t ldb, int64_t stride_b, T beta, T *c, int64_t ldc,
                              int64_t stride_c, int64_t batch_size,
                              const std::vector<sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_size);
    auto done = queue.submit([&](sycl::handler &cgh) {
        if (!verify_support<sycl::half, T>(queue, sycl::aspect::fp16)) {
            throw oneapi::mkl::unimplemented(
                "blas", "sycl::half", "half is not supported by the device or the sycl compiler");
        }
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType *>(a);
            auto b_ = reinterpret_cast<const cuDataType *>(b);
            auto c_ = reinterpret_cast<cuDataType *>(c);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_operation(transa),
                                     get_cublas_operation(transb), m, n, k, (cuDataType *)&alpha,
                                     a_, lda, stride_a, b_, ldb, stride_b, (cuDataType *)&beta, c_,
                                     ldc, stride_c, batch_size);
        });
    });
    return done;
}

#define GEMM_STRIDED_BATCH_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                     \
    sycl::event gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,     \
                           int64_t n, int64_t k, TYPE alpha, const TYPE *a, int64_t lda,          \
                           int64_t stride_a, const TYPE *b, int64_t ldb, int64_t stride_b,        \
                           TYPE beta, TYPE *c, int64_t ldc, int64_t stride_c, int64_t batch_size, \
                           const std::vector<sycl::event> &dependencies) {                        \
        return gemm_batch(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, \
                          a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, \
                          dependencies);                                                          \
    }

GEMM_STRIDED_BATCH_LAUNCHER_USM(sycl::half, cublasHgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(float, cublasSgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(double, cublasDgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, cublasCgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, cublasZgemmStridedBatched)

#undef GEMM_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event gemm_batch(const char *func_name, Func func, sycl::queue &queue,
                              transpose *transa, transpose *transb, int64_t *m, int64_t *n,
                              int64_t *k, T *alpha, const T **a, int64_t *lda, const T **b,
                              int64_t *ldb, T *beta, T **c, int64_t *ldc, int64_t group_count,
                              int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(m[i], n[i], k[i], lda[i], ldb[i], ldc[i], group_size[i]);
    }
    auto done = queue.submit([&](sycl::handler &cgh) {
        if (!verify_support<sycl::half, T>(queue, sycl::aspect::fp16)) {
            throw oneapi::mkl::unimplemented(
                "blas", "sycl::half", "half is not supported by the device or the sycl compiler");
        }
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            int64_t offset = 0;
            cublasStatus_t err;
            for (int64_t i = 0; i < group_count; i++) {
                auto **a_ = reinterpret_cast<const cuDataType **>(a);
                auto **b_ = reinterpret_cast<const cuDataType **>(b);
                auto **c_ = reinterpret_cast<cuDataType **>(c);
                CUBLAS_ERROR_FUNC_T_SYNC(
                    func_name, func, err, handle, get_cublas_operation(transa[i]),
                    get_cublas_operation(transb[i]), (int)m[i], (int)n[i], (int)k[i],
                    (cuDataType *)&alpha[i], a_ + offset, (int)lda[i], b_ + offset, (int)ldb[i],
                    (cuDataType *)&beta[i], c_ + offset, (int)ldc[i], (int)group_size[i]);
                offset += group_size[i];
            }
        });
    });
    return done;
}

#define GEMM_BATCH_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                             \
    sycl::event gemm_batch(sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,  \
                           int64_t *n, int64_t *k, TYPE *alpha, const TYPE **a, int64_t *lda,     \
                           const TYPE **b, int64_t *ldb, TYPE *beta, TYPE **c, int64_t *ldc,      \
                           int64_t group_count, int64_t *group_size,                              \
                           const std::vector<sycl::event> &dependencies) {                        \
        return gemm_batch(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, \
                          a, lda, b, ldb, beta, c, ldc, group_count, group_size, dependencies);   \
    }

GEMM_BATCH_LAUNCHER_USM(sycl::half, cublasHgemmBatched)
GEMM_BATCH_LAUNCHER_USM(float, cublasSgemmBatched)
GEMM_BATCH_LAUNCHER_USM(double, cublasDgemmBatched)
GEMM_BATCH_LAUNCHER_USM(std::complex<float>, cublasCgemmBatched)
GEMM_BATCH_LAUNCHER_USM(std::complex<double>, cublasZgemmBatched)

#undef GEMM_BATCH_LAUNCHER_USM

sycl::event trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, float alpha, const float *a,
                       int64_t lda, int64_t stride_a, float *b, int64_t ldb, int64_t stride_b,
                       int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

sycl::event trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, double alpha, const double *a,
                       int64_t lda, int64_t stride_a, double *b, int64_t ldb, int64_t stride_b,
                       int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

sycl::event trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                       const std::complex<float> *a, int64_t lda, int64_t stride_a,
                       std::complex<float> *b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

sycl::event trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                       const std::complex<double> *a, int64_t lda, int64_t stride_a,
                       std::complex<double> *b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

template <typename Func, typename T>
inline sycl::event trsm_batch(const char *func_name, Func func, sycl::queue &queue,
                              side *left_right, uplo *upper_lower, transpose *trans,
                              diag *unit_diag, int64_t *m, int64_t *n, T *alpha, const T **a,
                              int64_t *lda, T **b, int64_t *ldb, int64_t group_count,
                              int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(m[i], n[i], lda[i], ldb[i], group_size[i]);
    }
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            int64_t offset = 0;
            cublasStatus_t err;
            for (int64_t i = 0; i < group_count; i++) {
                auto **a_ = reinterpret_cast<const cuDataType **>(a);
                auto **b_ = reinterpret_cast<cuDataType **>(b);
                CUBLAS_ERROR_FUNC_T_SYNC(
                    func_name, func, err, handle, get_cublas_side_mode(left_right[i]),
                    get_cublas_fill_mode(upper_lower[i]), get_cublas_operation(trans[i]),
                    get_cublas_diag_type(unit_diag[i]), (int)m[i], (int)n[i],
                    (cuDataType *)&alpha[i], a_ + offset, (int)lda[i], b_ + offset, (int)ldb[i],
                    (int)group_size[i]);
                offset += group_size[i];
            }
        });
    });
    return done;
}

#define TRSM_BATCH_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                              \
    sycl::event trsm_batch(sycl::queue &queue, side *left_right, uplo *upper_lower,                \
                           transpose *trans, diag *unit_diag, int64_t *m, int64_t *n, TYPE *alpha, \
                           const TYPE **a, int64_t *lda, TYPE **b, int64_t *ldb,                   \
                           int64_t group_count, int64_t *group_size,                               \
                           const std::vector<sycl::event> &dependencies) {                         \
        return trsm_batch(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, left_right, upper_lower, trans,  \
                          unit_diag, m, n, alpha, a, lda, b, ldb, group_count, group_size,         \
                          dependencies);                                                           \
    }

TRSM_BATCH_LAUNCHER_USM(float, cublasStrsmBatched)
TRSM_BATCH_LAUNCHER_USM(double, cublasDtrsmBatched)
TRSM_BATCH_LAUNCHER_USM(std::complex<float>, cublasCtrsmBatched)
TRSM_BATCH_LAUNCHER_USM(std::complex<double>, cublasZtrsmBatched)

#undef TRSM_BATCH_LAUNCHER_USM

sycl::event syrk_batch(sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                       int64_t *k, float *alpha, const float **a, int64_t *lda, float *beta,
                       float **c, int64_t *ldc, int64_t group_count, int64_t *groupsize,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

sycl::event syrk_batch(sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                       int64_t *k, double *alpha, const double **a, int64_t *lda, double *beta,
                       double **c, int64_t *ldc, int64_t group_count, int64_t *groupsize,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

sycl::event syrk_batch(sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                       int64_t *k, std::complex<float> *alpha, const std::complex<float> **a,
                       int64_t *lda, std::complex<float> *beta, std::complex<float> **c,
                       int64_t *ldc, int64_t group_count, int64_t *groupsize,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

sycl::event syrk_batch(sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                       int64_t *k, std::complex<double> *alpha, const std::complex<double> **a,
                       int64_t *lda, std::complex<double> *beta, std::complex<double> **c,
                       int64_t *ldc, int64_t group_count, int64_t *groupsize,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

sycl::event syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       float alpha, const float *a, int64_t lda, int64_t stride_a, float beta,
                       float *c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

sycl::event syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       double alpha, const double *a, int64_t lda, int64_t stride_a, double beta,
                       double *c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

sycl::event syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                       int64_t stride_a, std::complex<float> beta, std::complex<float> *c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

sycl::event syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                       int64_t stride_a, std::complex<double> beta, std::complex<double> *c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                           const float *a, int64_t lda, int64_t stride_a, float *b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                           const double *a, int64_t lda, int64_t stride_a, double *b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                           std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                           int64_t stride_a, std::complex<float> *b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                           std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                           int64_t stride_a, std::complex<double> *b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                           float *ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                           double *ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                           std::complex<float> alpha, std::complex<float> *ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                           std::complex<double> alpha, std::complex<double> *ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

sycl::event omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, float alpha, const float *a, int64_t lda, int64_t stride_a,
                          float beta, const float *b, int64_t ldb, int64_t stride_b, float *c,
                          int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
}

sycl::event omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, double alpha, const double *a, int64_t lda, int64_t stride_a,
                          double beta, const double *b, int64_t ldb, int64_t stride_b, double *c,
                          int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
}

sycl::event omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                          int64_t lda, int64_t stride_a, std::complex<float> beta,
                          const std::complex<float> *b, int64_t ldb, int64_t stride_b,
                          std::complex<float> *c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
}

sycl::event omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                          int64_t lda, int64_t stride_a, std::complex<double> beta,
                          const std::complex<double> *b, int64_t ldb, int64_t stride_b,
                          std::complex<double> *c, int64_t ldc, int64_t stride_c,
                          int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           float *alpha, const float **a, int64_t *lda, float **b, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           double *alpha, const double **a, int64_t *lda, double **b, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           std::complex<float> *alpha, const std::complex<float> **a, int64_t *lda,
                           std::complex<float> **b, int64_t *ldb, int64_t group_count,
                           int64_t *groupsize, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           std::complex<double> *alpha, const std::complex<double> **a,
                           int64_t *lda, std::complex<double> **b, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           float *alpha, float **ab, int64_t *lda, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           double *alpha, double **ab, int64_t *lda, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           std::complex<float> *alpha, std::complex<float> **ab, int64_t *lda,
                           int64_t *ldb, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           std::complex<double> *alpha, std::complex<double> **ab, int64_t *lda,
                           int64_t *ldb, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

} // namespace column_major
namespace row_major {

// Buffer APIs

void copy_batch(sycl::queue &queue, int64_t n, sycl::buffer<float, 1> &x, int64_t incx,
                int64_t stridex, sycl::buffer<float, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

void copy_batch(sycl::queue &queue, int64_t n, sycl::buffer<double, 1> &x, int64_t incx,
                int64_t stridex, sycl::buffer<double, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

void copy_batch(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<float>, 1> &x,
                int64_t incx, int64_t stridex, sycl::buffer<std::complex<float>, 1> &y,
                int64_t incy, int64_t stridey, int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

void copy_batch(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<double>, 1> &x,
                int64_t incx, int64_t stridex, sycl::buffer<std::complex<double>, 1> &y,
                int64_t incy, int64_t stridey, int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

void axpy_batch(sycl::queue &queue, int64_t n, float alpha, sycl::buffer<float, 1> &x, int64_t incx,
                int64_t stridex, sycl::buffer<float, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

void axpy_batch(sycl::queue &queue, int64_t n, double alpha, sycl::buffer<double, 1> &x,
                int64_t incx, int64_t stridex, sycl::buffer<double, 1> &y, int64_t incy,
                int64_t stridey, int64_t batch_size) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

void axpy_batch(sycl::queue &queue, int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1> &x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<float>, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

void axpy_batch(sycl::queue &queue, int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1> &x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<double>, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

void gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n, float alpha,
                sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a, sycl::buffer<float, 1> &x,
                int64_t incx, int64_t stride_x, float beta, sycl::buffer<float, 1> &y, int64_t incy,
                int64_t stride_y, int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

void gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n, double alpha,
                sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<double, 1> &x, int64_t incx, int64_t stride_x, double beta,
                sycl::buffer<double, 1> &y, int64_t incy, int64_t stride_y, int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

void gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
                int64_t stride_x, std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &y,
                int64_t incy, int64_t stride_y, int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

void gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
                int64_t stride_x, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1> &y, int64_t incy, int64_t stride_y,
                int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

void dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a, sycl::buffer<float, 1> &x,
                int64_t incx, int64_t stride_x, sycl::buffer<float, 1> &c, int64_t ldc,
                int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

void dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<double, 1> &x, int64_t incx, int64_t stride_x,
                sycl::buffer<double, 1> &c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

void dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<std::complex<float>, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<float>, 1> &x, int64_t incx, int64_t stride_x,
                sycl::buffer<std::complex<float>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

void dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<std::complex<double>, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<double>, 1> &x, int64_t incx, int64_t stride_x,
                sycl::buffer<std::complex<double>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

template <typename Func, typename T>
inline void gemm_batch(const char *func_name, Func func, sycl::queue &queue, transpose transa,
                       transpose transb, int64_t m, int64_t n, int64_t k, T alpha,
                       sycl::buffer<T, 1> &a, int64_t lda, int64_t stride_a, sycl::buffer<T, 1> &b,
                       int64_t ldb, int64_t stride_b, T beta, sycl::buffer<T, 1> &c, int64_t ldc,
                       int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
}

#define GEMM_STRIDED_BATCH_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                          \
    void gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,  \
                    int64_t k, TYPE alpha, sycl::buffer<TYPE, 1> &a, int64_t lda,                  \
                    int64_t stride_a, sycl::buffer<TYPE, 1> &b, int64_t ldb, int64_t stride_b,     \
                    TYPE beta, sycl::buffer<TYPE, 1> &c, int64_t ldc, int64_t stride_c,            \
                    int64_t batch_size) {                                                          \
        gemm_batch(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, \
                   stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);                \
    }

GEMM_STRIDED_BATCH_LAUNCHER(sycl::half, cublasHgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER(float, cublasSgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER(double, cublasDgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER(std::complex<float>, cublasCgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER(std::complex<double>, cublasZgemmStridedBatched)

#undef GEMM_STRIDED_BATCH_LAUNCHER

void trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, float alpha, sycl::buffer<float, 1> &a,
                int64_t lda, int64_t stride_a, sycl::buffer<float, 1> &b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

void trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, double alpha, sycl::buffer<double, 1> &a,
                int64_t lda, int64_t stride_a, sycl::buffer<double, 1> &b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

void trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

void trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

void syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                float alpha, sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a, float beta,
                sycl::buffer<float, 1> &c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

void syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                double alpha, sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                double beta, sycl::buffer<double, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

void syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
                int64_t stride_a, std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &c,
                int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

void syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
                int64_t stride_a, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

void omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                    sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a,
                    sycl::buffer<float, 1> &b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

void omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                    sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                    sycl::buffer<double, 1> &b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

void omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
                    int64_t stride_a, sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
                    int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

void omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
                    int64_t lda, int64_t stride_a, sycl::buffer<std::complex<double>, 1> &b,
                    int64_t ldb, int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                    sycl::buffer<float, 1> &ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                    sycl::buffer<double, 1> &ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

void omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   float alpha, sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a,
                   float beta, sycl::buffer<float, 1> &b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<float, 1> &c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
}

void omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   double alpha, sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                   double beta, sycl::buffer<double, 1> &b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<double, 1> &c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
}

void omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
                   int64_t stride_a, std::complex<float> beta,
                   sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<std::complex<float>, 1> &c, int64_t ldc, int64_t stride_c,
                   int64_t batch_size) {
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
}

void omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
                   int64_t lda, int64_t stride_a, std::complex<double> beta,
                   sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<std::complex<double>, 1> &c, int64_t ldc, int64_t stride_c,
                   int64_t batch_size) {
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
}

// USM APIs

sycl::event copy_batch(sycl::queue &queue, int64_t *n, const float **x, int64_t *incx, float **y,
                       int64_t *incy, int64_t group_count, int64_t *group_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

sycl::event copy_batch(sycl::queue &queue, int64_t *n, const double **x, int64_t *incx, double **y,
                       int64_t *incy, int64_t group_count, int64_t *group_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

sycl::event copy_batch(sycl::queue &queue, int64_t *n, const std::complex<float> **x, int64_t *incx,
                       std::complex<float> **y, int64_t *incy, int64_t group_count,
                       int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

sycl::event copy_batch(sycl::queue &queue, int64_t *n, const std::complex<double> **x,
                       int64_t *incx, std::complex<double> **y, int64_t *incy, int64_t group_count,
                       int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

sycl::event copy_batch(sycl::queue &queue, int64_t n, const float *x, int64_t incx,
                       std::int64_t stridex, float *y, int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

sycl::event copy_batch(sycl::queue &queue, int64_t n, const double *x, int64_t incx,
                       std::int64_t stridex, double *y, int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

sycl::event copy_batch(sycl::queue &queue, int64_t n, const std::complex<float> *x, int64_t incx,
                       std::int64_t stridex, std::complex<float> *y, int64_t incy,
                       std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

sycl::event copy_batch(sycl::queue &queue, int64_t n, const std::complex<double> *x, int64_t incx,
                       std::int64_t stridex, std::complex<double> *y, int64_t incy,
                       std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

sycl::event axpy_batch(sycl::queue &queue, int64_t *n, float *alpha, const float **x, int64_t *incx,
                       float **y, int64_t *incy, int64_t group_count, int64_t *group_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

sycl::event axpy_batch(sycl::queue &queue, int64_t *n, double *alpha, const double **x,
                       int64_t *incx, double **y, int64_t *incy, int64_t group_count,
                       int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

sycl::event axpy_batch(sycl::queue &queue, int64_t *n, std::complex<float> *alpha,
                       const std::complex<float> **x, int64_t *incx, std::complex<float> **y,
                       int64_t *incy, int64_t group_count, int64_t *group_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

sycl::event axpy_batch(sycl::queue &queue, int64_t *n, std::complex<double> *alpha,
                       const std::complex<double> **x, int64_t *incx, std::complex<double> **y,
                       int64_t *incy, int64_t group_count, int64_t *group_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

sycl::event axpy_batch(sycl::queue &queue, int64_t n, float alpha, const float *x, int64_t incx,
                       int64_t stridex, float *y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

sycl::event axpy_batch(sycl::queue &queue, int64_t n, double alpha, const double *x, int64_t incx,
                       int64_t stridex, double *y, int64_t incy, int64_t stridey,
                       int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

sycl::event axpy_batch(sycl::queue &queue, int64_t n, std::complex<float> alpha,
                       const std::complex<float> *x, int64_t incx, int64_t stridex,
                       std::complex<float> *y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

sycl::event axpy_batch(sycl::queue &queue, int64_t n, std::complex<double> alpha,
                       const std::complex<double> *x, int64_t incx, int64_t stridex,
                       std::complex<double> *y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

sycl::event gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n, float alpha,
                       const float *a, int64_t lda, int64_t stride_a, const float *x, int64_t incx,
                       int64_t stride_x, float beta, float *y, int64_t incy, int64_t stride_y,
                       int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

sycl::event gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n, double alpha,
                       const double *a, int64_t lda, int64_t stride_a, const double *x,
                       int64_t incx, int64_t stride_x, double beta, double *y, int64_t incy,
                       int64_t stride_y, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

sycl::event gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                       std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                       int64_t stride_a, const std::complex<float> *x, int64_t incx,
                       int64_t stride_x, std::complex<float> beta, std::complex<float> *y,
                       int64_t incy, int64_t stride_y, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

sycl::event gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                       std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                       int64_t stride_a, const std::complex<double> *x, int64_t incx,
                       int64_t stride_x, std::complex<double> beta, std::complex<double> *y,
                       int64_t incy, int64_t stride_y, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

sycl::event gemv_batch(sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n, float *alpha,
                       const float **a, int64_t *lda, const float **x, int64_t *incx, float *beta,
                       float **y, int64_t *incy, int64_t group_count, int64_t *groupsize,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

sycl::event gemv_batch(sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n, double *alpha,
                       const double **a, int64_t *lda, const double **x, int64_t *incx,
                       double *beta, double **y, int64_t *incy, int64_t group_count,
                       int64_t *groupsize, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

sycl::event gemv_batch(sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n,
                       std::complex<float> *alpha, const std::complex<float> **a, int64_t *lda,
                       const std::complex<float> **x, int64_t *incx, std::complex<float> *beta,
                       std::complex<float> **y, int64_t *incy, int64_t group_count,
                       int64_t *groupsize, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

sycl::event gemv_batch(sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n,
                       std::complex<double> *alpha, const std::complex<double> **a, int64_t *lda,
                       const std::complex<double> **x, int64_t *incx, std::complex<double> *beta,
                       std::complex<double> **y, int64_t *incy, int64_t group_count,
                       int64_t *groupsize, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

sycl::event dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n, const float *a,
                       int64_t lda, int64_t stride_a, const float *x, int64_t incx,
                       int64_t stride_x, float *c, int64_t ldc, int64_t stride_c,
                       int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

sycl::event dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n, const double *a,
                       int64_t lda, int64_t stride_a, const double *x, int64_t incx,
                       int64_t stride_x, double *c, int64_t ldc, int64_t stride_c,
                       int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

sycl::event dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                       const std::complex<float> *a, int64_t lda, int64_t stride_a,
                       const std::complex<float> *x, int64_t incx, int64_t stride_x,
                       std::complex<float> *c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

sycl::event dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                       const std::complex<double> *a, int64_t lda, int64_t stride_a,
                       const std::complex<double> *x, int64_t incx, int64_t stride_x,
                       std::complex<double> *c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

sycl::event dgmm_batch(sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                       const float **a, int64_t *lda, const float **x, int64_t *incx, float **c,
                       int64_t *ldc, int64_t group_count, int64_t *groupsize,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

sycl::event dgmm_batch(sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                       const double **a, int64_t *lda, const double **x, int64_t *incx, double **c,
                       int64_t *ldc, int64_t group_count, int64_t *groupsize,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

sycl::event dgmm_batch(sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                       const std::complex<float> **a, int64_t *lda, const std::complex<float> **x,
                       int64_t *incx, std::complex<float> **c, int64_t *ldc, int64_t group_count,
                       int64_t *groupsize, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

sycl::event dgmm_batch(sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                       const std::complex<double> **a, int64_t *lda, const std::complex<double> **x,
                       int64_t *incx, std::complex<double> **c, int64_t *ldc, int64_t group_count,
                       int64_t *groupsize, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

template <typename Func, typename T>
inline sycl::event gemm_batch(const char *func_name, Func func, sycl::queue &queue,
                              transpose transa, transpose transb, int64_t m, int64_t n, int64_t k,
                              T alpha, const T *a, int64_t lda, int64_t stride_a, const T *b,
                              int64_t ldb, int64_t stride_b, T beta, T *c, int64_t ldc,
                              int64_t stride_c, int64_t batch_size,
                              const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
}

#define GEMM_STRIDED_BATCH_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                     \
    sycl::event gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,     \
                           int64_t n, int64_t k, TYPE alpha, const TYPE *a, int64_t lda,          \
                           int64_t stride_a, const TYPE *b, int64_t ldb, int64_t stride_b,        \
                           TYPE beta, TYPE *c, int64_t ldc, int64_t stride_c, int64_t batch_size, \
                           const std::vector<sycl::event> &dependencies) {                        \
        return gemm_batch(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, \
                          a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, \
                          dependencies);                                                          \
    }

GEMM_STRIDED_BATCH_LAUNCHER_USM(sycl::half, cublasHgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(float, cublasSgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(double, cublasDgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, cublasCgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, cublasZgemmStridedBatched)

#undef GEMM_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event gemm_batch(const char *func_name, Func func, sycl::queue &queue,
                              transpose *transa, transpose *transb, int64_t *m, int64_t *n,
                              int64_t *k, T *alpha, const T **a, int64_t *lda, const T **b,
                              int64_t *ldb, T *beta, T **c, int64_t *ldc, int64_t group_count,
                              int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
}

#define GEMM_BATCH_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                             \
    sycl::event gemm_batch(sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,  \
                           int64_t *n, int64_t *k, TYPE *alpha, const TYPE **a, int64_t *lda,     \
                           const TYPE **b, int64_t *ldb, TYPE *beta, TYPE **c, int64_t *ldc,      \
                           int64_t group_count, int64_t *group_size,                              \
                           const std::vector<sycl::event> &dependencies) {                        \
        return gemm_batch(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, \
                          a, lda, b, ldb, beta, c, ldc, group_count, group_size, dependencies);   \
    }

GEMM_BATCH_LAUNCHER_USM(sycl::half, cublasHgemmBatched)
GEMM_BATCH_LAUNCHER_USM(float, cublasSgemmBatched)
GEMM_BATCH_LAUNCHER_USM(double, cublasDgemmBatched)
GEMM_BATCH_LAUNCHER_USM(std::complex<float>, cublasCgemmBatched)
GEMM_BATCH_LAUNCHER_USM(std::complex<double>, cublasZgemmBatched)

#undef GEMM_BATCH_LAUNCHER_USM

sycl::event trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, float alpha, const float *a,
                       int64_t lda, int64_t stride_a, float *b, int64_t ldb, int64_t stride_b,
                       int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

sycl::event trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, double alpha, const double *a,
                       int64_t lda, int64_t stride_a, double *b, int64_t ldb, int64_t stride_b,
                       int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

sycl::event trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                       const std::complex<float> *a, int64_t lda, int64_t stride_a,
                       std::complex<float> *b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

sycl::event trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                       const std::complex<double> *a, int64_t lda, int64_t stride_a,
                       std::complex<double> *b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

template <typename Func, typename T>
inline sycl::event trsm_batch(const char *func_name, Func func, sycl::queue &queue,
                              side *left_right, uplo *upper_lower, transpose *trans,
                              diag *unit_diag, int64_t *m, int64_t *n, T *alpha, const T **a,
                              int64_t *lda, T **b, int64_t *ldb, int64_t group_count,
                              int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

#define TRSM_BATCH_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                              \
    sycl::event trsm_batch(sycl::queue &queue, side *left_right, uplo *upper_lower,                \
                           transpose *trans, diag *unit_diag, int64_t *m, int64_t *n, TYPE *alpha, \
                           const TYPE **a, int64_t *lda, TYPE **b, int64_t *ldb,                   \
                           int64_t group_count, int64_t *group_size,                               \
                           const std::vector<sycl::event> &dependencies) {                         \
        return trsm_batch(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, left_right, upper_lower, trans,  \
                          unit_diag, m, n, alpha, a, lda, b, ldb, group_count, group_size,         \
                          dependencies);                                                           \
    }

TRSM_BATCH_LAUNCHER_USM(float, cublasStrsmBatched)
TRSM_BATCH_LAUNCHER_USM(double, cublasDtrsmBatched)
TRSM_BATCH_LAUNCHER_USM(std::complex<float>, cublasCtrsmBatched)
TRSM_BATCH_LAUNCHER_USM(std::complex<double>, cublasZtrsmBatched)

#undef TRSM_BATCH_LAUNCHER_USM

sycl::event syrk_batch(sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                       int64_t *k, float *alpha, const float **a, int64_t *lda, float *beta,
                       float **c, int64_t *ldc, int64_t group_count, int64_t *groupsize,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

sycl::event syrk_batch(sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                       int64_t *k, double *alpha, const double **a, int64_t *lda, double *beta,
                       double **c, int64_t *ldc, int64_t group_count, int64_t *groupsize,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

sycl::event syrk_batch(sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                       int64_t *k, std::complex<float> *alpha, const std::complex<float> **a,
                       int64_t *lda, std::complex<float> *beta, std::complex<float> **c,
                       int64_t *ldc, int64_t group_count, int64_t *groupsize,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

sycl::event syrk_batch(sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                       int64_t *k, std::complex<double> *alpha, const std::complex<double> **a,
                       int64_t *lda, std::complex<double> *beta, std::complex<double> **c,
                       int64_t *ldc, int64_t group_count, int64_t *groupsize,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

sycl::event syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       float alpha, const float *a, int64_t lda, int64_t stride_a, float beta,
                       float *c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

sycl::event syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       double alpha, const double *a, int64_t lda, int64_t stride_a, double beta,
                       double *c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

sycl::event syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                       int64_t stride_a, std::complex<float> beta, std::complex<float> *c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

sycl::event syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                       int64_t stride_a, std::complex<double> beta, std::complex<double> *c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                           const float *a, int64_t lda, int64_t stride_a, float *b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                           const double *a, int64_t lda, int64_t stride_a, double *b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                           std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                           int64_t stride_a, std::complex<float> *b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                           std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                           int64_t stride_a, std::complex<double> *b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                           float *ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                           double *ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                           std::complex<float> alpha, std::complex<float> *ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                           std::complex<double> alpha, std::complex<double> *ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

sycl::event omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, float alpha, const float *a, int64_t lda, int64_t stride_a,
                          float beta, const float *b, int64_t ldb, int64_t stride_b, float *c,
                          int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
}

sycl::event omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, double alpha, const double *a, int64_t lda, int64_t stride_a,
                          double beta, const double *b, int64_t ldb, int64_t stride_b, double *c,
                          int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
}

sycl::event omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                          int64_t lda, int64_t stride_a, std::complex<float> beta,
                          const std::complex<float> *b, int64_t ldb, int64_t stride_b,
                          std::complex<float> *c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
}

sycl::event omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                          int64_t lda, int64_t stride_a, std::complex<double> beta,
                          const std::complex<double> *b, int64_t ldb, int64_t stride_b,
                          std::complex<double> *c, int64_t ldc, int64_t stride_c,
                          int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           float *alpha, const float **a, int64_t *lda, float **b, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           double *alpha, const double **a, int64_t *lda, double **b, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           std::complex<float> *alpha, const std::complex<float> **a, int64_t *lda,
                           std::complex<float> **b, int64_t *ldb, int64_t group_count,
                           int64_t *groupsize, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           std::complex<double> *alpha, const std::complex<double> **a,
                           int64_t *lda, std::complex<double> **b, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           float *alpha, float **ab, int64_t *lda, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           double *alpha, double **ab, int64_t *lda, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           std::complex<float> *alpha, std::complex<float> **ab, int64_t *lda,
                           int64_t *ldb, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           std::complex<double> *alpha, std::complex<double> **ab, int64_t *lda,
                           int64_t *ldb, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

} // namespace row_major
} // namespace cublas
} // namespace blas
} // namespace mkl
} // namespace oneapi
