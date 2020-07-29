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

// BLAS-like extensions

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
           std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
           std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
           cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    throw backend_unsupported_exception();
}

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
           std::int64_t n, std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
           std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
           cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    throw backend_unsupported_exception();
}

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
           std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    throw backend_unsupported_exception();
}

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
           std::int64_t n, std::int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc) {
    throw backend_unsupported_exception();
}

void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
              cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda, int8_t ao,
              cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb, uint8_t bo, float beta,
              cl::sycl::buffer<int32_t, 1> &c, std::int64_t ldc, cl::sycl::buffer<int32_t, 1> &co) {
    throw backend_unsupported_exception();
}

template <typename Func, typename T_A, typename T_B, typename T_C, typename DATATYPE_A,
          typename DATATYPE_B, typename DATATYPE_C>
inline void gemm_ext(Func func, DATATYPE_A DT_A, DATATYPE_B DT_B, DATATYPE_C DT_C,
                     cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                     int64_t n, int64_t k, T_C alpha, cl::sycl::buffer<T_A, 1> &a, int64_t lda,
                     cl::sycl::buffer<T_B, 1> &b, int64_t ldb, T_C beta,
                     cl::sycl::buffer<T_C, 1> &c, int64_t ldc) {
    using cuDataType_A = typename CudaEquivalentType<T_A>::Type;
    using cuDataType_B = typename CudaEquivalentType<T_B>::Type;
    using cuDataType_C = typename CudaEquivalentType<T_C>::Type;
    overflow_check(m, n, k, lda, ldb, ldc);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType_A *>(ih, a_acc);
            auto b_     = sc.get_mem<cuDataType_B *>(ih, b_acc);
            auto c_     = sc.get_mem<cuDataType_C *>(ih, c_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_operation(transa),
                              get_cublas_operation(transb), m, n, k, (cuDataType_C *)&alpha, a_,
                              DT_A, lda, b_, DT_B, ldb, (cuDataType_C *)&beta, c_, DT_C, ldc, DT_C,
                              CUBLAS_GEMM_DEFAULT);
        });
    });
}

#define GEMM_EXT_LAUNCHER(TYPE_A, TYPE_B, TYPE_C, CUBLAS_ROUTINE, CUDADATATYPE_A, CUDADATATYPE_B,  \
                          CUDADATATYPE_C)                                                          \
    void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,           \
                  int64_t n, int64_t k, TYPE_C alpha, cl::sycl::buffer<TYPE_A, 1> &a, int64_t lda, \
                  cl::sycl::buffer<TYPE_B, 1> &b, int64_t ldb, TYPE_C beta,                        \
                  cl::sycl::buffer<TYPE_C, 1> &c, int64_t ldc) {                                   \
        gemm_ext(CUBLAS_ROUTINE, CUDADATATYPE_A, CUDADATATYPE_B, CUDADATATYPE_C, queue, transa,    \
                 transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);                            \
    }

GEMM_EXT_LAUNCHER(half, half, float, cublasGemmEx, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F)
GEMM_EXT_LAUNCHER(half, half, half, cublasGemmEx, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F)
GEMM_EXT_LAUNCHER(float, float, float, cublasGemmEx, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F)
GEMM_EXT_LAUNCHER(double, double, double, cublasGemmEx, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F)
GEMM_EXT_LAUNCHER(std::complex<float>, std::complex<float>, std::complex<float>, cublasGemmEx,
                  CUDA_C_32F, CUDA_C_32F, CUDA_C_32F)
GEMM_EXT_LAUNCHER(std::complex<double>, std::complex<double>, std::complex<double>, cublasGemmEx,
                  CUDA_C_64F, CUDA_C_64F, CUDA_C_64F)

#undef GEMM_EXT_LAUNCHER

// USM APIs

// BLAS-like extensions

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      int64_t n, int64_t k, float alpha, const float *a, int64_t lda,
                      const float *b, int64_t ldb, float beta, float *c, int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw backend_unsupported_exception();
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      int64_t n, int64_t k, double alpha, const double *a, int64_t lda,
                      const double *b, int64_t ldb, double beta, double *c, int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw backend_unsupported_exception();
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      int64_t n, int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                      int64_t lda, const std::complex<float> *b, int64_t ldb,
                      std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw backend_unsupported_exception();
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      int64_t n, int64_t k, std::complex<double> alpha,
                      const std::complex<double> *a, int64_t lda, const std::complex<double> *b,
                      int64_t ldb, std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw backend_unsupported_exception();
}

} // namespace cublas
} // namespace mkl
} // namespace oneapi
