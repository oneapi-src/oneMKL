/*******************************************************************************
* Copyright Codeplay Software
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

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "portblas_common.hpp"
#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/blas/detail/portblas/onemkl_blas_portblas.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace portblas {
namespace column_major {

constexpr bool is_column_major() {
    return true;
}

// BUFFER
void gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, sycl::half alpha,
          sycl::buffer<sycl::half, 1> &a, std::int64_t lda, sycl::buffer<sycl::half, 1> &b,
          std::int64_t ldb, sycl::half beta, sycl::buffer<sycl::half, 1> &c, std::int64_t ldc) {
#ifdef ENABLE_PORTBLAS_HALF
    CALL_PORTBLAS_FN(::blas::_gemm, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                     ldc);
#else
    throw unimplemented("blas", "gemm", " half");
#endif
}

void gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
          sycl::buffer<sycl::half, 1> &a, std::int64_t lda, sycl::buffer<sycl::half, 1> &b,
          std::int64_t ldb, float beta, sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifdef ENABLE_PORTBLAS_HALF
    CALL_PORTBLAS_FN(::blas::_gemm, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                     ldc);
#else
    throw unimplemented("blas", "gemm", " for different argument data types");
#endif
}

// USM
sycl::event gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                 std::int64_t m, std::int64_t n, std::int64_t k, sycl::half alpha,
                 const sycl::half *a, std::int64_t lda, const sycl::half *b, std::int64_t ldb,
                 sycl::half beta, sycl::half *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies) {
#ifdef ENABLE_PORTBLAS_HALF
    CALL_PORTBLAS_USM_FN(::blas::_gemm, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                         c, ldc, dependencies);
#else
    throw unimplemented("blas", "gemm", " for USM");
#endif
}

sycl::event gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                 std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const sycl::half *a,
                 std::int64_t lda, const sycl::half *b, std::int64_t ldb, float beta, float *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
#ifdef ENABLE_PORTBLAS_HALF
    CALL_PORTBLAS_USM_FN(::blas::_gemm, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                         c, ldc, dependencies);
#else
    throw unimplemented("blas", "gemm", " for USM");
#endif
}
} // namespace column_major

namespace row_major {

constexpr bool is_column_major() {
    return false;
}

// BUFFER
void gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, sycl::half alpha,
          sycl::buffer<sycl::half, 1> &a, std::int64_t lda, sycl::buffer<sycl::half, 1> &b,
          std::int64_t ldb, sycl::half beta, sycl::buffer<sycl::half, 1> &c, std::int64_t ldc) {
#ifdef ENABLE_PORTBLAS_HALF
    CALL_PORTBLAS_FN(::blas::_gemm, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                     ldc);
#else
    throw unimplemented("blas", "gemm", " half");
#endif
}

void gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
          sycl::buffer<sycl::half, 1> &a, std::int64_t lda, sycl::buffer<sycl::half, 1> &b,
          std::int64_t ldb, float beta, sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifdef ENABLE_PORTBLAS_HALF
    CALL_PORTBLAS_FN(::blas::_gemm, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                     ldc);
#else
    throw unimplemented("blas", "gemm", " for different argument data types");
#endif
}

// USM
sycl::event gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                 std::int64_t m, std::int64_t n, std::int64_t k, sycl::half alpha,
                 const sycl::half *a, std::int64_t lda, const sycl::half *b, std::int64_t ldb,
                 sycl::half beta, sycl::half *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies) {
#ifdef ENABLE_PORTBLAS_HALF
    CALL_PORTBLAS_USM_FN(::blas::_gemm, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                         c, ldc, dependencies);
#else
    throw unimplemented("blas", "gemm", " for USM");
#endif
}

sycl::event gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                 std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const sycl::half *a,
                 std::int64_t lda, const sycl::half *b, std::int64_t ldb, float beta, float *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
#ifdef ENABLE_PORTBLAS_HALF
    CALL_PORTBLAS_USM_FN(::blas::_gemm, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                         c, ldc, dependencies);
#else
    throw unimplemented("blas", "gemm", " for USM");
#endif
}

} // namespace row_major
} // namespace portblas
} // namespace blas
} // namespace mkl
} // namespace oneapi
