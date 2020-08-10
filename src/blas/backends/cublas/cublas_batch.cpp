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
#include <CL/sycl/detail/pi.hpp>
#include "cublas_helper.hpp"
#include "cublas_scope_handle.hpp"
#include "include/exceptions_helper.hpp"
#include "oneapi/mkl/blas/detail/cublas/onemkl_blas_cublas.hpp"

namespace oneapi {
namespace mkl {
namespace cublas {

// Buffer APIs

template <typename Func, typename T>
inline void gemm_batch(Func func, cl::sycl::queue &queue, transpose transa, transpose transb,
                       int64_t m, int64_t n, int64_t k, T alpha, cl::sycl::buffer<T, 1> &a,
                       int64_t lda, int64_t stride_a, cl::sycl::buffer<T, 1> &b, int64_t ldb,
                       int64_t stride_b, T beta, cl::sycl::buffer<T, 1> &c, int64_t ldc,
                       int64_t stride_c, int64_t batch_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_size);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType *>(ih, a_acc);
            auto b_ = sc.get_mem<cuDataType *>(ih, b_acc);
            auto c_ = sc.get_mem<cuDataType *>(ih, c_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_operation(transa),
                              get_cublas_operation(transb), m, n, k, (cuDataType *)&alpha, a_, lda,
                              stride_a, b_, ldb, stride_b, (cuDataType *)&beta, c_, ldc, stride_c,
                              batch_size);
        });
    });
}

#define GEMM_STRIDED_BATCH_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                          \
    void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,         \
                    int64_t n, int64_t k, TYPE alpha, cl::sycl::buffer<TYPE, 1> &a, int64_t lda,   \
                    int64_t stride_a, cl::sycl::buffer<TYPE, 1> &b, int64_t ldb, int64_t stride_b, \
                    TYPE beta, cl::sycl::buffer<TYPE, 1> &c, int64_t ldc, int64_t stride_c,        \
                    int64_t batch_size) {                                                          \
        gemm_batch(CUBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b,     \
                   ldb, stride_b, beta, c, ldc, stride_c, batch_size);                             \
    }

GEMM_STRIDED_BATCH_LAUNCHER(float, cublasSgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER(double, cublasDgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER(std::complex<float>, cublasCgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER(std::complex<double>, cublasZgemmStridedBatched)

#undef GEMM_STRIDED_BATCH_LAUNCHER

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size) {
    throw backend_unsupported_exception();
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                cl::sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size) {
    throw backend_unsupported_exception();
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    throw backend_unsupported_exception();
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    throw backend_unsupported_exception();
}

// USM APIs

template <typename Func, typename T>
inline cl::sycl::event gemm_batch(Func func, cl::sycl::queue &queue, transpose transa,
                                  transpose transb, int64_t m, int64_t n, int64_t k, T alpha,
                                  const T *a, int64_t lda, int64_t stride_a, const T *b,
                                  int64_t ldb, int64_t stride_b, T beta, T *c, int64_t ldc,
                                  int64_t stride_c, int64_t batch_size,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_size);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType *>(a);
            auto b_ = reinterpret_cast<const cuDataType *>(b);
            auto c_ = reinterpret_cast<cuDataType *>(c);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_operation(transa),
                              get_cublas_operation(transb), m, n, k, (cuDataType *)&alpha, a_, lda,
                              stride_a, b_, ldb, stride_b, (cuDataType *)&beta, c_, ldc, stride_c,
                              batch_size);
        });
    });
    return done;
}

#define GEMM_STRIDED_BATCH_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                      \
    cl::sycl::event gemm_batch(                                                                    \
        cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,          \
        int64_t k, TYPE alpha, const TYPE *a, int64_t lda, int64_t stride_a, const TYPE *b,        \
        int64_t ldb, int64_t stride_b, TYPE beta, TYPE *c, int64_t ldc, int64_t stride_c,          \
        int64_t batch_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {         \
        return gemm_batch(CUBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, stride_a, \
                          b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, dependencies);     \
    }

GEMM_STRIDED_BATCH_LAUNCHER_USM(float, cublasSgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(double, cublasDgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, cublasCgemmStridedBatched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, cublasZgemmStridedBatched)

#undef GEMM_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline cl::sycl::event gemm_batch(Func func, cl::sycl::queue &queue, transpose *transa,
                                  transpose *transb, int64_t *m, int64_t *n, int64_t *k, T *alpha,
                                  const T **a, int64_t *lda, const T **b, int64_t *ldb, T *beta,
                                  T **c, int64_t *ldc, int64_t group_count, int64_t *group_size,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(m[i], n[i], k[i], lda[i], ldb[i], ldc[i], group_size[i]);
    }
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            int64_t offset = 0;
            cublasStatus_t err;
            for (int64_t i = 0; i < group_count; i++) {
                auto **a_ = reinterpret_cast<const cuDataType **>(a);
                auto **b_ = reinterpret_cast<const cuDataType **>(b);
                auto **c_ = reinterpret_cast<cuDataType **>(c);
                CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_operation(transa[i]),
                                  get_cublas_operation(transb[i]), (int)m[i], (int)n[i], (int)k[i],
                                  (cuDataType *)&alpha[i], a_ + offset, (int)lda[i], b_ + offset,
                                  (int)ldb[i], (cuDataType *)&beta[i], c_ + offset, (int)ldc[i],
                                  (int)group_size[i]);
                offset += group_size[i];
            }
        });
    });
    return done;
}

#define GEMM_BATCH_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                            \
    cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,     \
                               int64_t *m, int64_t *n, int64_t *k, TYPE *alpha, const TYPE **a,  \
                               int64_t *lda, const TYPE **b, int64_t *ldb, TYPE *beta, TYPE **c, \
                               int64_t *ldc, int64_t group_count, int64_t *group_size,           \
                               const cl::sycl::vector_class<cl::sycl::event> &dependencies) {    \
        return gemm_batch(CUBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, \
                          beta, c, ldc, group_count, group_size, dependencies);                  \
    }

GEMM_BATCH_LAUNCHER_USM(float, cublasSgemmBatched)
GEMM_BATCH_LAUNCHER_USM(double, cublasDgemmBatched)
GEMM_BATCH_LAUNCHER_USM(std::complex<float>, cublasCgemmBatched)
GEMM_BATCH_LAUNCHER_USM(std::complex<double>, cublasZgemmBatched)

#undef GEMM_BATCH_LAUNCHER_USM

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, float *alpha, const float **x,
                           int64_t *incx, float **y, int64_t *incy, int64_t group_count,
                           int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw backend_unsupported_exception();
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, double *alpha, const double **x,
                           int64_t *incx, double **y, int64_t *incy, int64_t group_count,
                           int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw backend_unsupported_exception();
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, std::complex<float> *alpha,
                           const std::complex<float> **x, int64_t *incx, std::complex<float> **y,
                           int64_t *incy, int64_t group_count, int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw backend_unsupported_exception();
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, std::complex<double> *alpha,
                           const std::complex<double> **x, int64_t *incx, std::complex<double> **y,
                           int64_t *incy, int64_t group_count, int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw backend_unsupported_exception();
}

} // namespace cublas
} // namespace mkl
} // namespace oneapi
