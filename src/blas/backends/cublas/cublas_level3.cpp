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
#include "onemkl/blas/detail/cublas/onemkl_blas_cublas.hpp"

namespace onemkl {
namespace cublas {
template <typename Func, typename T>
inline void gemm(Func func, cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                 int64_t n, int64_t k, T alpha, cl::sycl::buffer<T, 1> &a, int64_t lda,
                 cl::sycl::buffer<T, 1> &b, int64_t ldb, T beta, cl::sycl::buffer<T, 1> &c,
                 int64_t ldc) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, ldb, ldc);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto b_     = sc.get_mem<cuDataType *>(ih, b_acc);
            auto c_     = sc.get_mem<cuDataType *>(ih, c_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_operation(transa),
                              get_cublas_operation(transb), m, n, k, (cuDataType *)&alpha, a_, lda,
                              b_, ldb, (cuDataType *)&beta, c_, ldc);
        });
    });
}

#define GEMM_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                        \
    void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,    \
              int64_t k, TYPE alpha, cl::sycl::buffer<TYPE, 1> &a, int64_t lda,                    \
              cl::sycl::buffer<TYPE, 1> &b, int64_t ldb, TYPE beta, cl::sycl::buffer<TYPE, 1> &c,  \
              int64_t ldc) {                                                                       \
        gemm(CUBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); \
    }

GEMM_LAUNCHER(float, cublasSgemm)
GEMM_LAUNCHER(double, cublasDgemm)
GEMM_LAUNCHER(std::complex<float>, cublasCgemm)
GEMM_LAUNCHER(std::complex<double>, cublasZgemm)

#undef GEMM_LAUNCHER

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, half alpha, cl::sycl::buffer<half, 1> &a,
          std::int64_t lda, cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
          cl::sycl::buffer<half, 1> &c, std::int64_t ldc) {
    throw std::runtime_error("Not implemented for cublas");
}

template <typename Func, typename T>
inline void symm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,
                 int64_t n, T alpha, cl::sycl::buffer<T, 1> &a, int64_t lda,
                 cl::sycl::buffer<T, 1> &b, int64_t ldb, T beta, cl::sycl::buffer<T, 1> &c,
                 int64_t ldc) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, ldc);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto b_     = sc.get_mem<cuDataType *>(ih, b_acc);
            auto c_     = sc.get_mem<cuDataType *>(ih, c_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_side_mode(left_right),
                              get_cublas_fill_mode(upper_lower), m, n, (cuDataType *)&alpha, a_,
                              lda, b_, ldb, (cuDataType *)&beta, c_, ldc);
        });
    });
}

#define SYMM_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                        \
    void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,     \
              TYPE alpha, cl::sycl::buffer<TYPE, 1> &a, int64_t lda, cl::sycl::buffer<TYPE, 1> &b, \
              int64_t ldb, TYPE beta, cl::sycl::buffer<TYPE, 1> &c, int64_t ldc) {                 \
        symm(CUBLAS_ROUTINE, queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, \
             ldc);                                                                                 \
    }

SYMM_LAUNCHER(float, cublasSsymm)
SYMM_LAUNCHER(double, cublasDsymm)
SYMM_LAUNCHER(std::complex<float>, cublasCsymm)
SYMM_LAUNCHER(std::complex<double>, cublasZsymm)

#undef SYMM_LAUNCHER

template <typename Func, typename T>
inline void hemm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,
                 int64_t n, T alpha, cl::sycl::buffer<T, 1> &a, int64_t lda,
                 cl::sycl::buffer<T, 1> &b, int64_t ldb, T beta, cl::sycl::buffer<T, 1> &c,
                 int64_t ldc) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, ldc);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto b_     = sc.get_mem<cuDataType *>(ih, b_acc);
            auto c_     = sc.get_mem<cuDataType *>(ih, c_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_side_mode(left_right),
                              get_cublas_fill_mode(upper_lower), m, n, (cuDataType *)&alpha, a_,
                              lda, b_, ldb, (cuDataType *)&beta, c_, ldc);
        });
    });
}

#define HEMM_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                        \
    void hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,     \
              TYPE alpha, cl::sycl::buffer<TYPE, 1> &a, int64_t lda, cl::sycl::buffer<TYPE, 1> &b, \
              int64_t ldb, TYPE beta, cl::sycl::buffer<TYPE, 1> &c, int64_t ldc) {                 \
        hemm(CUBLAS_ROUTINE, queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, \
             ldc);                                                                                 \
    }
HEMM_LAUNCHER(std::complex<float>, cublasChemm)
HEMM_LAUNCHER(std::complex<double>, cublasZhemm)

#undef HEMM_LAUNCHER

template <typename Func, typename T>
inline void syrk(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                 int64_t k, T alpha, cl::sycl::buffer<T, 1> &a, int64_t lda, T beta,
                 cl::sycl::buffer<T, 1> &c, int64_t ldc) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, k, lda, ldc);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto c_     = sc.get_mem<cuDataType *>(ih, c_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower),
                              get_cublas_operation(trans), n, k, (cuDataType *)&alpha, a_, lda,
                              (cuDataType *)&beta, c_, ldc);
        });
    });
}

#define SYRK_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                    \
    void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, \
              TYPE alpha, cl::sycl::buffer<TYPE, 1> &a, int64_t lda, TYPE beta,                \
              cl::sycl::buffer<TYPE, 1> &c, int64_t ldc) {                                     \
        syrk(CUBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);    \
    }

SYRK_LAUNCHER(float, cublasSsyrk)
SYRK_LAUNCHER(double, cublasDsyrk)
SYRK_LAUNCHER(std::complex<float>, cublasCsyrk)
SYRK_LAUNCHER(std::complex<double>, cublasZsyrk)

#undef SYRK_LAUNCHER

template <typename Func, typename DataType, typename ScalarType>
inline void herk(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                 int64_t k, ScalarType alpha, cl::sycl::buffer<DataType, 1> &a, int64_t lda,
                 ScalarType beta, cl::sycl::buffer<DataType, 1> &c, int64_t ldc) {
    using cuDataType   = typename CudaEquivalentType<DataType>::Type;
    using cuScalarType = typename CudaEquivalentType<ScalarType>::Type;
    overflow_check(n, k, lda, ldc);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto c_     = sc.get_mem<cuDataType *>(ih, c_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower),
                              get_cublas_operation(trans), n, k, (cuScalarType *)&alpha, a_, lda,
                              (cuScalarType *)&beta, c_, ldc);
        });
    });
}

#define HERK_LAUNCHER(DATA_TYPE, SCALAR_TYPE, CUBLAS_ROUTINE)                                      \
    void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,     \
              SCALAR_TYPE alpha, cl::sycl::buffer<DATA_TYPE, 1> &a, int64_t lda, SCALAR_TYPE beta, \
              cl::sycl::buffer<DATA_TYPE, 1> &c, int64_t ldc) {                                    \
        herk(CUBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);        \
    }

HERK_LAUNCHER(std::complex<float>, float, cublasCherk)
HERK_LAUNCHER(std::complex<double>, double, cublasZherk)

#undef HERK_LAUNCHER

template <typename Func, typename T>
inline void syr2k(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                  int64_t k, T alpha, cl::sycl::buffer<T, 1> &a, int64_t lda,
                  cl::sycl::buffer<T, 1> &b, int64_t ldb, T beta, cl::sycl::buffer<T, 1> &c,
                  int64_t ldc) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, k, lda, ldb, ldc);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto b_     = sc.get_mem<cuDataType *>(ih, b_acc);
            auto c_     = sc.get_mem<cuDataType *>(ih, c_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower),
                              get_cublas_operation(trans), n, k, (cuDataType *)&alpha, a_, lda, b_,
                              ldb, (cuDataType *)&beta, c_, ldc);
        });
    });
}

#define SYR2K_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                       \
    void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,    \
               TYPE alpha, cl::sycl::buffer<TYPE, 1> &a, int64_t lda,                              \
               cl::sycl::buffer<TYPE, 1> &b, int64_t ldb, TYPE beta, cl::sycl::buffer<TYPE, 1> &c, \
               int64_t ldc) {                                                                      \
        syr2k(CUBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c,     \
              ldc);                                                                                \
    }
SYR2K_LAUNCHER(float, cublasSsyr2k)
SYR2K_LAUNCHER(double, cublasDsyr2k)
SYR2K_LAUNCHER(std::complex<float>, cublasCsyr2k)
SYR2K_LAUNCHER(std::complex<double>, cublasZsyr2k)

#undef SYR2K_LAUNCHER

template <typename Func, typename DataType, typename ScalarType>
inline void her2k(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                  int64_t k, DataType alpha, cl::sycl::buffer<DataType, 1> &a, int64_t lda,
                  cl::sycl::buffer<DataType, 1> &b, int64_t ldb, ScalarType beta,
                  cl::sycl::buffer<DataType, 1> &c, int64_t ldc) {
    using cuDataType   = typename CudaEquivalentType<DataType>::Type;
    using cuScalarType = typename CudaEquivalentType<ScalarType>::Type;
    overflow_check(n, k, lda, ldb, ldc);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto b_     = sc.get_mem<cuDataType *>(ih, b_acc);
            auto c_     = sc.get_mem<cuDataType *>(ih, c_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower),
                              get_cublas_operation(trans), n, k, (cuDataType *)&alpha, a_, lda, b_,
                              ldb, (cuScalarType *)&beta, c_, ldc);
        });
    });
}

#define HER2K_LAUNCHER(DATA_TYPE, SCALAR_TYPE, CUBLAS_ROUTINE)                                  \
    void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, \
               DATA_TYPE alpha, cl::sycl::buffer<DATA_TYPE, 1> &a, int64_t lda,                 \
               cl::sycl::buffer<DATA_TYPE, 1> &b, int64_t ldb, SCALAR_TYPE beta,                \
               cl::sycl::buffer<DATA_TYPE, 1> &c, int64_t ldc) {                                \
        her2k(CUBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c,  \
              ldc);                                                                             \
    }

HER2K_LAUNCHER(std::complex<float>, float, cublasCher2k)
HER2K_LAUNCHER(std::complex<double>, double, cublasZher2k)

#undef HER2K_LAUNCHER

// NOTE: In cublas TRMM diverted from the netlib blas and for performance
// reason it requires the C matrix to be
// separated from the B matrix. It is possible to use B instead of C, but this
// will slow-down the code.
template <typename Func, typename T>
inline void trmm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                 transpose trans, diag unit_diag, int64_t m, int64_t n, T alpha,
                 cl::sycl::buffer<T, 1> &a, int64_t lda, cl::sycl::buffer<T, 1> &b, int64_t ldb) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto b_     = sc.get_mem<cuDataType *>(ih, b_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_side_mode(left_right),
                              get_cublas_fill_mode(upper_lower), get_cublas_operation(trans),
                              get_cublas_diag_type(unit_diag), m, n, (cuDataType *)&alpha, a_, lda,
                              b_, ldb, b_, ldb);
        });
    });
}

#define TRMM_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                    \
    void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,      \
              diag unit_diag, int64_t m, int64_t n, TYPE alpha, cl::sycl::buffer<TYPE, 1> &a,  \
              int64_t lda, cl::sycl::buffer<TYPE, 1> &b, int64_t ldb) {                        \
        trmm(CUBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, \
             lda, b, ldb);                                                                     \
    }
TRMM_LAUNCHER(float, cublasStrmm)
TRMM_LAUNCHER(double, cublasDtrmm)
TRMM_LAUNCHER(std::complex<float>, cublasCtrmm)
TRMM_LAUNCHER(std::complex<double>, cublasZtrmm)

#undef TRMM_LAUNCHER

template <typename Func, typename T>
inline void trsm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                 transpose trans, diag unit_diag, int64_t m, int64_t n, T alpha,
                 cl::sycl::buffer<T, 1> &a, int64_t lda, cl::sycl::buffer<T, 1> &b, int64_t ldb) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto b_     = sc.get_mem<cuDataType *>(ih, b_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_side_mode(left_right),
                              get_cublas_fill_mode(upper_lower), get_cublas_operation(trans),
                              get_cublas_diag_type(unit_diag), m, n, (cuDataType *)&alpha, a_, lda,
                              b_, ldb);
        });
    });
}

#define TRSM_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                    \
    void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,      \
              diag unit_diag, int64_t m, int64_t n, TYPE alpha, cl::sycl::buffer<TYPE, 1> &a,  \
              int64_t lda, cl::sycl::buffer<TYPE, 1> &b, int64_t ldb) {                        \
        trsm(CUBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, \
             lda, b, ldb);                                                                     \
    }
TRSM_LAUNCHER(float, cublasStrsm)
TRSM_LAUNCHER(double, cublasDtrsm)
TRSM_LAUNCHER(std::complex<float>, cublasCtrsm)
TRSM_LAUNCHER(std::complex<double>, cublasZtrsm)

#undef TRSM_LAUNCHER
} // namespace cublas
} // namespace onemkl
