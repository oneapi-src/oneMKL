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

void gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<int8_t, 1> &a, int64_t lda,
               int8_t ao, sycl::buffer<int8_t, 1> &b, int64_t ldb, int8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, int64_t ldc, sycl::buffer<int32_t, 1> &co) {
    throw unimplemented("blas", "gemm_bias", "for column_major layout");
}

void gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<int8_t, 1> &a, int64_t lda,
               int8_t ao, sycl::buffer<uint8_t, 1> &b, int64_t ldb, uint8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, int64_t ldc, sycl::buffer<int32_t, 1> &co) {
    throw unimplemented("blas", "gemm_bias", "for column_major layout");
}

void gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<uint8_t, 1> &a, int64_t lda,
               uint8_t ao, sycl::buffer<int8_t, 1> &b, int64_t ldb, int8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, int64_t ldc, sycl::buffer<int32_t, 1> &co) {
    throw unimplemented("blas", "gemm_bias", "for column_major layout");
}

void gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<uint8_t, 1> &a, int64_t lda,
               uint8_t ao, sycl::buffer<uint8_t, 1> &b, int64_t ldb, uint8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, int64_t ldc, sycl::buffer<int32_t, 1> &co) {
    throw unimplemented("blas", "gemm_bias", "for column_major layout");
}

void gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, float alpha, sycl::buffer<float, 1> &a, int64_t lda,
           sycl::buffer<float, 1> &b, int64_t ldb, float beta, sycl::buffer<float, 1> &c,
           int64_t ldc) {
    throw unimplemented("blas", "gemmt", "for column_major layout");
}

void gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, double alpha, sycl::buffer<double, 1> &a, int64_t lda,
           sycl::buffer<double, 1> &b, int64_t ldb, double beta, sycl::buffer<double, 1> &c,
           int64_t ldc) {
    throw unimplemented("blas", "gemmt", "for column_major layout");
}

void gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a,
           int64_t lda, sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
           std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    throw unimplemented("blas", "gemmt", "for column_major layout");
}

void gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
           int64_t lda, sycl::buffer<std::complex<double>, 1> &b, int64_t ldb,
           std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    throw unimplemented("blas", "gemmt", "for column_major layout");
}

void omatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
              sycl::buffer<float, 1> &a, int64_t lda, sycl::buffer<float, 1> &b, int64_t ldb) {
    throw unimplemented("blas", "omatcopy", "for column_major layout");
}

void omatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
              sycl::buffer<double, 1> &a, int64_t lda, sycl::buffer<double, 1> &b, int64_t ldb) {
    throw unimplemented("blas", "omatcopy", "for column_major layout");
}

void omatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, std::complex<float> alpha,
              sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
              sycl::buffer<std::complex<float>, 1> &b, int64_t ldb) {
    throw unimplemented("blas", "omatcopy", "for column_major layout");
}

void omatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, std::complex<double> alpha,
              sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
              sycl::buffer<std::complex<double>, 1> &b, int64_t ldb) {
    throw unimplemented("blas", "omatcopy", "for column_major layout");
}

void imatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
              sycl::buffer<float, 1> &ab, int64_t lda, int64_t ldb) {
    throw unimplemented("blas", "imatcopy", "for column_major layout");
}

void imatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
              sycl::buffer<double, 1> &ab, int64_t lda, int64_t ldb) {
    throw unimplemented("blas", "imatcopy", "for column_major layout");
}

void imatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, std::complex<float> alpha,
              sycl::buffer<std::complex<float>, 1> &ab, int64_t lda, int64_t ldb) {
    throw unimplemented("blas", "imatcopy", "for column_major layout");
}

void imatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, std::complex<double> alpha,
              sycl::buffer<std::complex<double>, 1> &ab, int64_t lda, int64_t ldb) {
    throw unimplemented("blas", "imatcopy", "for column_major layout");
}

void omatadd(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
             float alpha, sycl::buffer<float, 1> &a, int64_t lda, float beta,
             sycl::buffer<float, 1> &b, int64_t ldb, sycl::buffer<float, 1> &c, int64_t ldc) {
    throw unimplemented("blas", "omatadd", "for column_major layout");
}

void omatadd(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
             double alpha, sycl::buffer<double, 1> &a, int64_t lda, double beta,
             sycl::buffer<double, 1> &b, int64_t ldb, sycl::buffer<double, 1> &c, int64_t ldc) {
    throw unimplemented("blas", "omatadd", "for column_major layout");
}

void omatadd(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
             std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
             std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
             sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    throw unimplemented("blas", "omatadd", "for column_major layout");
}

void omatadd(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
             std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
             std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &b, int64_t ldb,
             sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    throw unimplemented("blas", "omatadd", "for column_major layout");
}

// USM APIs

sycl::event gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const int8_t *a, int64_t lda,
                      int8_t ao, const int8_t *b, int64_t ldb, int8_t bo, float beta, int32_t *c,
                      int64_t ldc, const int32_t *co,
                      const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm_bias", "for column_major layout");
}

sycl::event gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const int8_t *a, int64_t lda,
                      int8_t ao, const uint8_t *b, int64_t ldb, uint8_t bo, float beta, int32_t *c,
                      int64_t ldc, const int32_t *co,
                      const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm_bias", "for column_major layout");
}

sycl::event gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const uint8_t *a, int64_t lda,
                      uint8_t ao, const int8_t *b, int64_t ldb, int8_t bo, float beta, int32_t *c,
                      int64_t ldc, const int32_t *co,
                      const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm_bias", "for column_major layout");
}

sycl::event gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const uint8_t *a, int64_t lda,
                      uint8_t ao, const uint8_t *b, int64_t ldb, uint8_t bo, float beta, int32_t *c,
                      int64_t ldc, const int32_t *co,
                      const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm_bias", "for column_major layout");
}

sycl::event gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, float alpha, const float *a, int64_t lda, const float *b,
                  int64_t ldb, float beta, float *c, int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemmt", "for column_major layout");
}

sycl::event gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, double alpha, const double *a, int64_t lda, const double *b,
                  int64_t ldb, double beta, double *c, int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemmt", "for column_major layout");
}

sycl::event gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                  int64_t lda, const std::complex<float> *b, int64_t ldb, std::complex<float> beta,
                  std::complex<float> *c, int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemmt", "for column_major layout");
}

sycl::event gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                  int64_t lda, const std::complex<double> *b, int64_t ldb,
                  std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemmt", "for column_major layout");
}

sycl::event omatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                     const float *a, int64_t lda, float *b, int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy", "for column_major layout");
}

sycl::event omatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                     const double *a, int64_t lda, double *b, int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy", "for column_major layout");
}

sycl::event omatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                     std::complex<float> *b, int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy", "for column_major layout");
}

sycl::event omatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                     std::complex<double> *b, int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy", "for column_major layout");
}

sycl::event imatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                     float *ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy", "for column_major layout");
}

sycl::event imatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                     double *ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy", "for column_major layout");
}

sycl::event imatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                     std::complex<float> alpha, std::complex<float> *ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy", "for column_major layout");
}

sycl::event imatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                     std::complex<double> alpha, std::complex<double> *ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy", "for column_major layout");
}

sycl::event omatadd(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                    float alpha, const float *a, int64_t lda, float beta, const float *b,
                    int64_t ldb, float *c, int64_t ldc,
                    const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatadd", "for column_major layout");
}

sycl::event omatadd(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                    double alpha, const double *a, int64_t lda, double beta, const double *b,
                    int64_t ldb, double *c, int64_t ldc,
                    const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatadd", "for column_major layout");
}

sycl::event omatadd(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                    std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                    std::complex<float> beta, const std::complex<float> *b, int64_t ldb,
                    std::complex<float> *c, int64_t ldc,
                    const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatadd", "for column_major layout");
}

sycl::event omatadd(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                    std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                    std::complex<double> beta, const std::complex<double> *b, int64_t ldb,
                    std::complex<double> *c, int64_t ldc,
                    const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatadd", "for column_major layout");
}

} // namespace column_major


namespace row_major {

// Buffer APIs

void gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<int8_t, 1> &a, int64_t lda,
               int8_t ao, sycl::buffer<int8_t, 1> &b, int64_t ldb, int8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, int64_t ldc, sycl::buffer<int32_t, 1> &co) {
    throw unimplemented("blas", "gemm_bias", "for row_major layout");
}

void gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<int8_t, 1> &a, int64_t lda,
               int8_t ao, sycl::buffer<uint8_t, 1> &b, int64_t ldb, uint8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, int64_t ldc, sycl::buffer<int32_t, 1> &co) {
    throw unimplemented("blas", "gemm_bias", "for row_major layout");
}

void gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<uint8_t, 1> &a, int64_t lda,
               uint8_t ao, sycl::buffer<int8_t, 1> &b, int64_t ldb, int8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, int64_t ldc, sycl::buffer<int32_t, 1> &co) {
    throw unimplemented("blas", "gemm_bias", "for row_major layout");
}

void gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<uint8_t, 1> &a, int64_t lda,
               uint8_t ao, sycl::buffer<uint8_t, 1> &b, int64_t ldb, uint8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, int64_t ldc, sycl::buffer<int32_t, 1> &co) {
    throw unimplemented("blas", "gemm_bias", "for row_major layout");
}

void gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, float alpha, sycl::buffer<float, 1> &a, int64_t lda,
           sycl::buffer<float, 1> &b, int64_t ldb, float beta, sycl::buffer<float, 1> &c,
           int64_t ldc) {
    throw unimplemented("blas", "gemmt", "for row_major layout");
}

void gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, double alpha, sycl::buffer<double, 1> &a, int64_t lda,
           sycl::buffer<double, 1> &b, int64_t ldb, double beta, sycl::buffer<double, 1> &c,
           int64_t ldc) {
    throw unimplemented("blas", "gemmt", "for row_major layout");
}

void gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a,
           int64_t lda, sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
           std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    throw unimplemented("blas", "gemmt", "for row_major layout");
}

void gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
           int64_t lda, sycl::buffer<std::complex<double>, 1> &b, int64_t ldb,
           std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    throw unimplemented("blas", "gemmt", "for row_major layout");
}

void omatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
              sycl::buffer<float, 1> &a, int64_t lda, sycl::buffer<float, 1> &b, int64_t ldb) {
    overflow_check(m, n, lda, ldb);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.get_access<sycl::access::mode::write>(cgh);
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<const float *>(a_acc);
            auto b_ = sc.get_mem<float *>(b_acc);
            const float beta = 0.f;
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC_SYNC(cublasSgeam, err, handle, get_cublas_operation(trans),
                                   CUBLAS_OP_N, m, n, &alpha,
                                   a_, lda, &beta, a_, ldb, b_, ldb);
       });
    });
}

void omatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
              sycl::buffer<double, 1> &a, int64_t lda, sycl::buffer<double, 1> &b, int64_t ldb) {
    overflow_check(m, n, lda, ldb);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.get_access<sycl::access::mode::write>(cgh);
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<const double *>(a_acc);
            auto b_ = sc.get_mem<double *>(b_acc);
            const double beta = 0.0;
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC_SYNC(cublasDgeam, err, handle, get_cublas_operation(trans),
                                   CUBLAS_OP_N, m, n, &alpha,
                                   a_, lda, &beta, a_, ldb, b_, ldb);
       });
    });
}

void omatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, std::complex<float> alpha,
              sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
              sycl::buffer<std::complex<float>, 1> &b, int64_t ldb) {
    using cuDataType = typename CudaEquivalentType<std::complex<float>>::Type;
    overflow_check(m, n, lda, ldb);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.get_access<sycl::access::mode::write>(cgh);
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<const cuDataType *>(a_acc);
            auto b_ = sc.get_mem<cuDataType *>(b_acc);
            const cuDataType beta {0.f, 0.f};
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC_SYNC(cublasCgeam, err, handle, get_cublas_operation(trans),
                                   CUBLAS_OP_N, m, n, (cuDataType *)&alpha,
                                   a_, lda, &beta, a_, ldb, b_, ldb);
       });
    });
}

void omatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, std::complex<double> alpha,
              sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
              sycl::buffer<std::complex<double>, 1> &b, int64_t ldb) {
    using cuDataType = typename CudaEquivalentType<std::complex<double>>::Type;
    overflow_check(m, n, lda, ldb);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.get_access<sycl::access::mode::write>(cgh);
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<const cuDataType *>(a_acc);
            auto b_ = sc.get_mem<cuDataType *>(b_acc);
            const cuDataType beta {0.0, 0.0};
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC_SYNC(cublasZgeam, err, handle, get_cublas_operation(trans),
                                   CUBLAS_OP_N, m, n, (cuDataType *)&alpha,
                                   a_, lda, &beta, a_, ldb, b_, ldb);
       });
    });
}

void imatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
              sycl::buffer<float, 1> &ab, int64_t lda, int64_t ldb) {
    throw unimplemented("blas", "imatcopy", "for row_major layout");
}

void imatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
              sycl::buffer<double, 1> &ab, int64_t lda, int64_t ldb) {
    throw unimplemented("blas", "imatcopy", "for row_major layout");
}

void imatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, std::complex<float> alpha,
              sycl::buffer<std::complex<float>, 1> &ab, int64_t lda, int64_t ldb) {
    throw unimplemented("blas", "imatcopy", "for row_major layout");
}

void imatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, std::complex<double> alpha,
              sycl::buffer<std::complex<double>, 1> &ab, int64_t lda, int64_t ldb) {
    throw unimplemented("blas", "imatcopy", "for row_major layout");
}

void omatadd(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
             float alpha, sycl::buffer<float, 1> &a, int64_t lda, float beta,
             sycl::buffer<float, 1> &b, int64_t ldb, sycl::buffer<float, 1> &c, int64_t ldc) {
    throw unimplemented("blas", "omatadd", "for row_major layout");
}

void omatadd(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
             double alpha, sycl::buffer<double, 1> &a, int64_t lda, double beta,
             sycl::buffer<double, 1> &b, int64_t ldb, sycl::buffer<double, 1> &c, int64_t ldc) {
    throw unimplemented("blas", "omatadd", "for row_major layout");
}

void omatadd(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
             std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
             std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
             sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    throw unimplemented("blas", "omatadd", "for row_major layout");
}

void omatadd(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
             std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
             std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &b, int64_t ldb,
             sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    throw unimplemented("blas", "omatadd", "for row_major layout");
}

// USM APIs

sycl::event gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const int8_t *a, int64_t lda,
                      int8_t ao, const int8_t *b, int64_t ldb, int8_t bo, float beta, int32_t *c,
                      int64_t ldc, const int32_t *co,
                      const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm_bias", "for row_major layout");
}

sycl::event gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const int8_t *a, int64_t lda,
                      int8_t ao, const uint8_t *b, int64_t ldb, uint8_t bo, float beta, int32_t *c,
                      int64_t ldc, const int32_t *co,
                      const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm_bias", "for row_major layout");
}

sycl::event gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const uint8_t *a, int64_t lda,
                      uint8_t ao, const int8_t *b, int64_t ldb, int8_t bo, float beta, int32_t *c,
                      int64_t ldc, const int32_t *co,
                      const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm_bias", "for row_major layout");
}

sycl::event gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const uint8_t *a, int64_t lda,
                      uint8_t ao, const uint8_t *b, int64_t ldb, uint8_t bo, float beta, int32_t *c,
                      int64_t ldc, const int32_t *co,
                      const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm_bias", "for row_major layout");
}

sycl::event gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, float alpha, const float *a, int64_t lda, const float *b,
                  int64_t ldb, float beta, float *c, int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemmt", "for row_major layout");
}

sycl::event gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, double alpha, const double *a, int64_t lda, const double *b,
                  int64_t ldb, double beta, double *c, int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemmt", "for row_major layout");
}

sycl::event gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                  int64_t lda, const std::complex<float> *b, int64_t ldb, std::complex<float> beta,
                  std::complex<float> *c, int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemmt", "for row_major layout");
}

sycl::event gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                  int64_t lda, const std::complex<double> *b, int64_t ldb,
                  std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemmt", "for row_major layout");
}

sycl::event omatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                     const float *a, int64_t lda, float *b, int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    overflow_check(m, n, lda, ldb);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            const float beta = 0.f;
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC_SYNC(cublasSgeam, err, handle, get_cublas_operation(trans),
                                   CUBLAS_OP_N, m, n, &alpha,
                                   a, lda, &beta, a, ldb, b, ldb);
       });
    });
    return done;
}

sycl::event omatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                     const double *a, int64_t lda, double *b, int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    overflow_check(m, n, lda, ldb);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            const double beta = 0.0;
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC_SYNC(cublasDgeam, err, handle, get_cublas_operation(trans),
                                   CUBLAS_OP_N, m, n, &alpha,
                                   a, lda, &beta, a, ldb, b, ldb);
       });
    });
    return done;
}

sycl::event omatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                     std::complex<float> *b, int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<std::complex<float>>::Type;
    overflow_check(m, n, lda, ldb);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType *>(a);
            auto b_ = reinterpret_cast<cuDataType *>(b);
            const cuDataType beta {0.f, 0.f};
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC_SYNC(cublasCgeam, err, handle, get_cublas_operation(trans),
                                   CUBLAS_OP_N, m, n, (cuDataType *)&alpha,
                                   a_, lda, &beta, a_, ldb, b_, ldb);
       });
    });
    return done;
}

sycl::event omatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                     std::complex<double> *b, int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<std::complex<double>>::Type;
    overflow_check(m, n, lda, ldb);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType *>(a);
            auto b_ = reinterpret_cast<cuDataType *>(b);
            const cuDataType beta {0.0, 0.0};
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC_SYNC(cublasZgeam, err, handle, get_cublas_operation(trans),
                                   CUBLAS_OP_N, m, n, (cuDataType *)&alpha,
                                   a_, lda, &beta, a_, ldb, b_, ldb);
       });
    });
    return done;
}

sycl::event imatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                     float *ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy", "for row_major layout");
}

sycl::event imatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                     double *ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy", "for row_major layout");
}

sycl::event imatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                     std::complex<float> alpha, std::complex<float> *ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy", "for row_major layout");
}

sycl::event imatcopy(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                     std::complex<double> alpha, std::complex<double> *ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy", "for row_major layout");
}

sycl::event omatadd(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                    float alpha, const float *a, int64_t lda, float beta, const float *b,
                    int64_t ldb, float *c, int64_t ldc,
                    const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatadd", "for row_major layout");
}

sycl::event omatadd(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                    double alpha, const double *a, int64_t lda, double beta, const double *b,
                    int64_t ldb, double *c, int64_t ldc,
                    const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatadd", "for row_major layout");
}

sycl::event omatadd(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                    std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                    std::complex<float> beta, const std::complex<float> *b, int64_t ldb,
                    std::complex<float> *c, int64_t ldc,
                    const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatadd", "for row_major layout");
}

sycl::event omatadd(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                    std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                    std::complex<double> beta, const std::complex<double> *b, int64_t ldb,
                    std::complex<double> *c, int64_t ldc,
                    const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatadd", "for row_major layout");
}

} // namespace row_major
} // namespace cublas
} // namespace blas
} // namespace mkl
} // namespace oneapi
