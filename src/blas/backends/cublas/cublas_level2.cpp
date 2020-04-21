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
inline void gemv(Func func, cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, T alpha,
                 cl::sycl::buffer<T, 1> &a, int64_t lda, cl::sycl::buffer<T, 1> &x, int64_t incx,
                 T beta, cl::sycl::buffer<T, 1> &y, int64_t incy) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, m, lda, incx, incy);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            auto y_     = sc.get_mem<cuDataType *>(ih, y_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_operation(trans), m, n,
                              (cuDataType *)&alpha, a_, lda, x_, incx, (cuDataType *)&beta, y_,
                              incy);
        });
    });
}

#define GEMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                              \
    void gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, TYPE alpha, \
              cl::sycl::buffer<TYPE, 1> &a, int64_t lda, cl::sycl::buffer<TYPE, 1> &x,   \
              int64_t incx, TYPE beta, cl::sycl::buffer<TYPE, 1> &y, int64_t incy) {     \
        gemv(CUBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy); \
    }

GEMV_LAUNCHER(float, cublasSgemv)
GEMV_LAUNCHER(double, cublasDgemv)
GEMV_LAUNCHER(std::complex<float>, cublasCgemv)
GEMV_LAUNCHER(std::complex<double>, cublasZgemv)
#undef GEMV_LAUNCHER

template <typename Func, typename T>
inline void gbmv(Func func, cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                 int64_t kl, int64_t ku, T alpha, cl::sycl::buffer<T, 1> &a, int64_t lda,
                 cl::sycl::buffer<T, 1> &x, int64_t incx, T beta, cl::sycl::buffer<T, 1> &y,
                 int64_t incy) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, m, lda, kl, ku, incx, incy);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            auto y_     = sc.get_mem<cuDataType *>(ih, y_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_operation(trans), m, n, kl, ku,
                              (cuDataType *)&alpha, a_, lda, x_, incx, (cuDataType *)&beta, y_,
                              incy);
        });
    });
}

#define GBMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                        \
    void gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl,           \
              int64_t ku, TYPE alpha, cl::sycl::buffer<TYPE, 1> &a, int64_t lda,                   \
              cl::sycl::buffer<TYPE, 1> &x, int64_t incx, TYPE beta, cl::sycl::buffer<TYPE, 1> &y, \
              int64_t incy) {                                                                      \
        gbmv(CUBLAS_ROUTINE, queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);   \
    }

GBMV_LAUNCHER(float, cublasSgbmv)
GBMV_LAUNCHER(double, cublasDgbmv)
GBMV_LAUNCHER(std::complex<float>, cublasCgbmv)
GBMV_LAUNCHER(std::complex<double>, cublasZgbmv)
#undef GBMV_LAUNCHER

template <typename Func, typename T>
inline void ger(Func func, cl::sycl::queue &queue, int64_t m, int64_t n, T alpha,
                cl::sycl::buffer<T, 1> &x, int64_t incx, cl::sycl::buffer<T, 1> &y, int64_t incy,
                cl::sycl::buffer<T, 1> &a, int64_t lda) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, m, lda, incx, incy);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<cl::sycl::access::mode::read>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            auto y_     = sc.get_mem<cuDataType *>(ih, y_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, m, n, (cuDataType *)&alpha, x_, incx, y_, incy, a_,
                              lda);
        });
    });
}

#define GER_LAUNCHER(EXT, TYPE, CUBLAS_ROUTINE)                                             \
    void ger##EXT(cl::sycl::queue &queue, int64_t m, int64_t n, TYPE alpha,                 \
                  cl::sycl::buffer<TYPE, 1> &x, int64_t incx, cl::sycl::buffer<TYPE, 1> &y, \
                  int64_t incy, cl::sycl::buffer<TYPE, 1> &a, int64_t lda) {                \
        ger(CUBLAS_ROUTINE, queue, m, n, alpha, x, incx, y, incy, a, lda);                  \
    }

GER_LAUNCHER(, float, cublasSger)
GER_LAUNCHER(, double, cublasDger)
GER_LAUNCHER(u, std::complex<float>, cublasCgeru)
GER_LAUNCHER(u, std::complex<double>, cublasZgeru)
GER_LAUNCHER(c, std::complex<float>, cublasCgerc)
GER_LAUNCHER(c, std::complex<double>, cublasZgerc)
#undef GER_LAUNCHER

template <typename Func, typename T>
inline void hbmv(Func func, cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, T alpha,
                 cl::sycl::buffer<T, 1> &a, int64_t lda, cl::sycl::buffer<T, 1> &x, int64_t incx,
                 T beta, cl::sycl::buffer<T, 1> &y, int64_t incy) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx, incy);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            auto y_     = sc.get_mem<cuDataType *>(ih, y_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower), n, k,
                              (cuDataType *)&alpha, a_, lda, x_, incx, (cuDataType *)&beta, y_,
                              incy);
        });
    });
}

#define HBMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                    \
    void hbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,      \
              cl::sycl::buffer<TYPE, 1> &a, int64_t lda, cl::sycl::buffer<TYPE, 1> &x,         \
              int64_t incx, TYPE beta, cl::sycl::buffer<TYPE, 1> &y, int64_t incy) {           \
        hbmv(CUBLAS_ROUTINE, queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy); \
    }

HBMV_LAUNCHER(std::complex<float>, cublasChbmv)
HBMV_LAUNCHER(std::complex<double>, cublasZhbmv)
#undef HBMV_LAUNCHER

template <typename Func, typename T>
inline void hemv(Func func, cl::sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 cl::sycl::buffer<T, 1> &a, int64_t lda, cl::sycl::buffer<T, 1> &x, int64_t incx,
                 T beta, cl::sycl::buffer<T, 1> &y, int64_t incy) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            auto y_     = sc.get_mem<cuDataType *>(ih, y_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower), n,
                              (cuDataType *)&alpha, a_, lda, x_, incx, (cuDataType *)&beta, y_,
                              incy);
        });
    });
}

#define HEMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                 \
    void hemv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,              \
              cl::sycl::buffer<TYPE, 1> &a, int64_t lda, cl::sycl::buffer<TYPE, 1> &x,      \
              int64_t incx, TYPE beta, cl::sycl::buffer<TYPE, 1> &y, int64_t incy) {        \
        hemv(CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy); \
    }

HEMV_LAUNCHER(std::complex<float>, cublasChemv)
HEMV_LAUNCHER(std::complex<double>, cublasZhemv)
#undef HEMV_LAUNCHER

template <typename Func, typename ScalarType, typename DataType>
inline void her(Func func, cl::sycl::queue &queue, uplo upper_lower, int64_t n, ScalarType alpha,
                cl::sycl::buffer<DataType, 1> &x, int64_t incx, cl::sycl::buffer<DataType, 1> &a,
                int64_t lda) {
    using cuScalarType = typename CudaEquivalentType<ScalarType>::Type;
    using cuDataType   = typename CudaEquivalentType<DataType>::Type;
    overflow_check(n, lda, incx);

    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower), n,
                              (cuScalarType *)&alpha, x_, incx, a_, lda);
        });
    });
}

#define HER_LAUNCHER(SCALAR_TYPE, DATA_TYPE, CUBLAS_ROUTINE)                                     \
    void her(cl::sycl::queue &queue, uplo upper_lower, int64_t n, SCALAR_TYPE alpha,             \
             cl::sycl::buffer<DATA_TYPE, 1> &x, int64_t incx, cl::sycl::buffer<DATA_TYPE, 1> &a, \
             int64_t lda) {                                                                      \
        her(CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, lda);                      \
    }

HER_LAUNCHER(float, std::complex<float>, cublasCher)
HER_LAUNCHER(double, std::complex<double>, cublasZher)

#undef HER_LAUNCHER

template <typename Func, typename T>
inline void her2(Func func, cl::sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 cl::sycl::buffer<T, 1> &x, int64_t incx, cl::sycl::buffer<T, 1> &y, int64_t incy,
                 cl::sycl::buffer<T, 1> &a, int64_t lda) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<cl::sycl::access::mode::read>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            auto y_     = sc.get_mem<cuDataType *>(ih, y_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower), n,
                              (cuDataType *)&alpha, x_, incx, y_, incy, a_, lda);
        });
    });
}

#define HER2_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                             \
    void her2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,          \
              cl::sycl::buffer<TYPE, 1> &x, int64_t incx, cl::sycl::buffer<TYPE, 1> &y, \
              int64_t incy, cl::sycl::buffer<TYPE, 1> &a, int64_t lda) {                \
        her2(CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);   \
    }

HER2_LAUNCHER(std::complex<float>, cublasCher2)
HER2_LAUNCHER(std::complex<double>, cublasZher2)

#undef HER2_LAUNCHER

template <typename Func, typename T>
inline void hpmv(Func func, cl::sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 cl::sycl::buffer<T, 1> &a, cl::sycl::buffer<T, 1> &x, int64_t incx, T beta,
                 cl::sycl::buffer<T, 1> &y, int64_t incy) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            auto y_     = sc.get_mem<cuDataType *>(ih, y_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower), n,
                              (cuDataType *)&alpha, a_, x_, incx, (cuDataType *)&beta, y_, incy);
        });
    });
}

#define HPMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                        \
    void hpmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                     \
              cl::sycl::buffer<TYPE, 1> &a, cl::sycl::buffer<TYPE, 1> &x, int64_t incx, TYPE beta, \
              cl::sycl::buffer<TYPE, 1> &y, int64_t incy) {                                        \
        hpmv(CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);             \
    }

HPMV_LAUNCHER(std::complex<float>, cublasChpmv)
HPMV_LAUNCHER(std::complex<double>, cublasZhpmv)

#undef HPMV_LAUNCHER

template <typename Func, typename ScalarType, typename DataType>
inline void hpr(Func func, cl::sycl::queue &queue, uplo upper_lower, int64_t n, ScalarType alpha,
                cl::sycl::buffer<DataType, 1> &x, int64_t incx, cl::sycl::buffer<DataType, 1> &a) {
    using cuScalarType = typename CudaEquivalentType<ScalarType>::Type;
    using cuDataType   = typename CudaEquivalentType<DataType>::Type;
    overflow_check(n, incx);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower), n,
                              (cuScalarType *)&alpha, x_, incx, a_);
        });
    });
}

#define HPR_LAUNCHER(SCALAR_TYPE, DATA_TYPE, CUBLAS_ROUTINE)                                       \
    void hpr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, SCALAR_TYPE alpha,               \
             cl::sycl::buffer<DATA_TYPE, 1> &x, int64_t incx, cl::sycl::buffer<DATA_TYPE, 1> &a) { \
        hpr(CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a);                             \
    }

HPR_LAUNCHER(float, std::complex<float>, cublasChpr)
HPR_LAUNCHER(double, std::complex<double>, cublasZhpr)

#undef HPR_LAUNCHER

template <typename Func, typename T>
inline void hpr2(Func func, cl::sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 cl::sycl::buffer<T, 1> &x, int64_t incx, cl::sycl::buffer<T, 1> &y, int64_t incy,
                 cl::sycl::buffer<T, 1> &a) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<cl::sycl::access::mode::read>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            auto y_     = sc.get_mem<cuDataType *>(ih, y_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower), n,
                              (cuDataType *)&alpha, x_, incx, y_, incy, a_);
        });
    });
}

#define HPR2_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                             \
    void hpr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,          \
              cl::sycl::buffer<TYPE, 1> &x, int64_t incx, cl::sycl::buffer<TYPE, 1> &y, \
              int64_t incy, cl::sycl::buffer<TYPE, 1> &a) {                             \
        hpr2(CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a);        \
    }

HPR2_LAUNCHER(std::complex<float>, cublasChpr2)
HPR2_LAUNCHER(std::complex<double>, cublasZhpr2)

#undef HPR2_LAUNCHER

template <typename Func, typename T>
inline void sbmv(Func func, cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, T alpha,
                 cl::sycl::buffer<T, 1> &a, int64_t lda, cl::sycl::buffer<T, 1> &x, int64_t incx,
                 T beta, cl::sycl::buffer<T, 1> &y, int64_t incy) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx, incy);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            auto y_     = sc.get_mem<cuDataType *>(ih, y_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower), n, k,
                              (cuDataType *)&alpha, a_, lda, x_, incx, (cuDataType *)&beta, y_,
                              incy);
        });
    });
}

#define SBMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                    \
    void sbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,      \
              cl::sycl::buffer<TYPE, 1> &a, int64_t lda, cl::sycl::buffer<TYPE, 1> &x,         \
              int64_t incx, TYPE beta, cl::sycl::buffer<TYPE, 1> &y, int64_t incy) {           \
        sbmv(CUBLAS_ROUTINE, queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy); \
    }

SBMV_LAUNCHER(float, cublasSsbmv)
SBMV_LAUNCHER(double, cublasDsbmv)

#undef SBMV_LAUNCHER

template <typename Func, typename T>
inline void symv(Func func, cl::sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 cl::sycl::buffer<T, 1> &a, int64_t lda, cl::sycl::buffer<T, 1> &x, int64_t incx,
                 T beta, cl::sycl::buffer<T, 1> &y, int64_t incy) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            auto y_     = sc.get_mem<cuDataType *>(ih, y_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower), n,
                              (cuDataType *)&alpha, a_, lda, x_, incx, (cuDataType *)&beta, y_,
                              incy);
        });
    });
}

#define SYMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                 \
    void symv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,              \
              cl::sycl::buffer<TYPE, 1> &a, int64_t lda, cl::sycl::buffer<TYPE, 1> &x,      \
              int64_t incx, TYPE beta, cl::sycl::buffer<TYPE, 1> &y, int64_t incy) {        \
        symv(CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy); \
    }

SYMV_LAUNCHER(float, cublasSsymv)
SYMV_LAUNCHER(double, cublasDsymv)

#undef SYMV_LAUNCHER

template <typename Func, typename T>
inline void syr(Func func, cl::sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                cl::sycl::buffer<T, 1> &x, int64_t incx, cl::sycl::buffer<T, 1> &a, int64_t lda) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower), n,
                              (cuDataType *)&alpha, x_, incx, a_, lda);
        });
    });
}

#define SYR_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                             \
    void syr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,          \
             cl::sycl::buffer<TYPE, 1> &x, int64_t incx, cl::sycl::buffer<TYPE, 1> &a, \
             int64_t lda) {                                                            \
        syr(CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, lda);            \
    }

SYR_LAUNCHER(float, cublasSsyr)
SYR_LAUNCHER(double, cublasDsyr)
// Intel does not support the following two
SYR_LAUNCHER(std::complex<float>, cublasCsyr)
SYR_LAUNCHER(std::complex<double>, cublasZsyr)
#undef SYR_LAUNCHER

template <typename Func, typename T>
inline void syr2(Func func, cl::sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 cl::sycl::buffer<T, 1> &x, int64_t incx, cl::sycl::buffer<T, 1> &y, int64_t incy,
                 cl::sycl::buffer<T, 1> &a, int64_t lda) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<cl::sycl::access::mode::read>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            auto y_     = sc.get_mem<cuDataType *>(ih, y_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower), n,
                              (cuDataType *)&alpha, x_, incx, y_, incy, a_, lda);
        });
    });
}

#define SYR2_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                             \
    void syr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,          \
              cl::sycl::buffer<TYPE, 1> &x, int64_t incx, cl::sycl::buffer<TYPE, 1> &y, \
              int64_t incy, cl::sycl::buffer<TYPE, 1> &a, int64_t lda) {                \
        syr2(CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);   \
    }

SYR2_LAUNCHER(float, cublasSsyr2)
SYR2_LAUNCHER(double, cublasDsyr2)
// Intel does not support the following two
SYR2_LAUNCHER(std::complex<float>, cublasCsyr2)
SYR2_LAUNCHER(std::complex<double>, cublasZsyr2)

#undef SYR2_LAUNCHER

template <typename Func, typename T>
inline void spmv(Func func, cl::sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 cl::sycl::buffer<T, 1> &a, cl::sycl::buffer<T, 1> &x, int64_t incx, T beta,
                 cl::sycl::buffer<T, 1> &y, int64_t incy) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            auto y_     = sc.get_mem<cuDataType *>(ih, y_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower), n,
                              (cuDataType *)&alpha, a_, x_, incx, (cuDataType *)&beta, y_, incy);
        });
    });
}

#define SPMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                        \
    void spmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                     \
              cl::sycl::buffer<TYPE, 1> &a, cl::sycl::buffer<TYPE, 1> &x, int64_t incx, TYPE beta, \
              cl::sycl::buffer<TYPE, 1> &y, int64_t incy) {                                        \
        spmv(CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);             \
    }

SPMV_LAUNCHER(float, cublasSspmv)
SPMV_LAUNCHER(double, cublasDspmv)

#undef SPMV_LAUNCHER

template <typename Func, typename T>
inline void spr(Func func, cl::sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                cl::sycl::buffer<T, 1> &x, int64_t incx, cl::sycl::buffer<T, 1> &a) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower), n,
                              (cuDataType *)&alpha, x_, incx, a_);
        });
    });
}

#define SPR_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                               \
    void spr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,            \
             cl::sycl::buffer<TYPE, 1> &x, int64_t incx, cl::sycl::buffer<TYPE, 1> &a) { \
        spr(CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a);                   \
    }

SPR_LAUNCHER(float, cublasSspr)
SPR_LAUNCHER(double, cublasDspr)

#undef SPR_LAUNCHER

template <typename Func, typename T>
inline void spr2(Func func, cl::sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 cl::sycl::buffer<T, 1> &x, int64_t incx, cl::sycl::buffer<T, 1> &y, int64_t incy,
                 cl::sycl::buffer<T, 1> &a) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<cl::sycl::access::mode::read>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            auto y_     = sc.get_mem<cuDataType *>(ih, y_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower), n,
                              (cuDataType *)&alpha, x_, incx, y_, incy, a_);
        });
    });
}

#define SPR2_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                             \
    void spr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,          \
              cl::sycl::buffer<TYPE, 1> &x, int64_t incx, cl::sycl::buffer<TYPE, 1> &y, \
              int64_t incy, cl::sycl::buffer<TYPE, 1> &a) {                             \
        spr2(CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a);        \
    }

SPR2_LAUNCHER(float, cublasSspr2)
SPR2_LAUNCHER(double, cublasDspr2)

#undef SPR2_LAUNCHER

template <typename Func, typename T>
inline void tbmv(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                 diag unit_diag, int64_t n, int64_t k, cl::sycl::buffer<T, 1> &a, int64_t lda,
                 cl::sycl::buffer<T, 1> &x, int64_t incx) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower),
                              get_cublas_operation(trans), get_cublas_diag_type(unit_diag), n, k,
                              a_, lda, x_, incx);
        });
    });
}

#define TBMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                \
    void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,   \
              int64_t n, int64_t k, cl::sycl::buffer<TYPE, 1> &a, int64_t lda,             \
              cl::sycl::buffer<TYPE, 1> &x, int64_t incx) {                                \
        tbmv(CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx); \
    }

TBMV_LAUNCHER(float, cublasStbmv)
TBMV_LAUNCHER(double, cublasDtbmv)
TBMV_LAUNCHER(std::complex<float>, cublasCtbmv)
TBMV_LAUNCHER(std::complex<double>, cublasZtbmv)

#undef TBMV_LAUNCHER

template <typename Func, typename T>
inline void tbsv(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                 diag unit_diag, int64_t n, int64_t k, cl::sycl::buffer<T, 1> &a, int64_t lda,
                 cl::sycl::buffer<T, 1> &x, int64_t incx) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower),
                              get_cublas_operation(trans), get_cublas_diag_type(unit_diag), n, k,
                              a_, lda, x_, incx);
        });
    });
}

#define TBSV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                \
    void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,   \
              int64_t n, int64_t k, cl::sycl::buffer<TYPE, 1> &a, int64_t lda,             \
              cl::sycl::buffer<TYPE, 1> &x, int64_t incx) {                                \
        tbsv(CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx); \
    }

TBSV_LAUNCHER(float, cublasStbsv)
TBSV_LAUNCHER(double, cublasDtbsv)
TBSV_LAUNCHER(std::complex<float>, cublasCtbsv)
TBSV_LAUNCHER(std::complex<double>, cublasZtbsv)

#undef TBSV_LAUNCHER

template <typename Func, typename T>
inline void tpmv(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                 diag unit_diag, int64_t n, cl::sycl::buffer<T, 1> &a, cl::sycl::buffer<T, 1> &x,
                 int64_t incx) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower),
                              get_cublas_operation(trans), get_cublas_diag_type(unit_diag), n, a_,
                              x_, incx);
        });
    });
}

#define TPMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                              \
    void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, \
              int64_t n, cl::sycl::buffer<TYPE, 1> &a, cl::sycl::buffer<TYPE, 1> &x,     \
              int64_t incx) {                                                            \
        tpmv(CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, x, incx);       \
    }

TPMV_LAUNCHER(float, cublasStpmv)
TPMV_LAUNCHER(double, cublasDtpmv)
TPMV_LAUNCHER(std::complex<float>, cublasCtpmv)
TPMV_LAUNCHER(std::complex<double>, cublasZtpmv)

#undef TPMV_LAUNCHER

template <typename Func, typename T>
inline void tpsv(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                 diag unit_diag, int64_t n, cl::sycl::buffer<T, 1> &a, cl::sycl::buffer<T, 1> &x,
                 int64_t incx) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower),
                              get_cublas_operation(trans), get_cublas_diag_type(unit_diag), n, a_,
                              x_, incx);
        });
    });
}

#define TPSV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                              \
    void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, \
              int64_t n, cl::sycl::buffer<TYPE, 1> &a, cl::sycl::buffer<TYPE, 1> &x,     \
              int64_t incx) {                                                            \
        tpsv(CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, x, incx);       \
    }

TPSV_LAUNCHER(float, cublasStpsv)
TPSV_LAUNCHER(double, cublasDtpsv)
TPSV_LAUNCHER(std::complex<float>, cublasCtpsv)
TPSV_LAUNCHER(std::complex<double>, cublasZtpsv)

#undef TPSV_LAUNCHER

template <typename Func, typename T>
inline void trmv(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                 diag unit_diag, int64_t n, cl::sycl::buffer<T, 1> &a, int64_t lda,
                 cl::sycl::buffer<T, 1> &x, int64_t incx) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower),
                              get_cublas_operation(trans), get_cublas_diag_type(unit_diag), n, a_,
                              lda, x_, incx);
        });
    });
}

#define TRMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                       \
    void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,          \
              int64_t n, cl::sycl::buffer<TYPE, 1> &a, int64_t lda, cl::sycl::buffer<TYPE, 1> &x, \
              int64_t incx) {                                                                     \
        trmv(CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);           \
    }

TRMV_LAUNCHER(float, cublasStrmv)
TRMV_LAUNCHER(double, cublasDtrmv)
TRMV_LAUNCHER(std::complex<float>, cublasCtrmv)
TRMV_LAUNCHER(std::complex<double>, cublasZtrmv)

#undef TRMV_LAUNCHER

template <typename Func, typename T>
inline void trsv(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                 diag unit_diag, int64_t n, cl::sycl::buffer<T, 1> &a, int64_t lda,
                 cl::sycl::buffer<T, 1> &x, int64_t incx) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.interop_task([=](cl::sycl::interop_handler ih) {
            auto sc     = CublasScopedContextHandler(queue);
            auto handle = sc.get_handle(queue);
            auto a_     = sc.get_mem<cuDataType *>(ih, a_acc);
            auto x_     = sc.get_mem<cuDataType *>(ih, x_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(upper_lower),
                              get_cublas_operation(trans), get_cublas_diag_type(unit_diag), n, a_,
                              lda, x_, incx);
        });
    });
}

#define TRSV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                       \
    void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,          \
              int64_t n, cl::sycl::buffer<TYPE, 1> &a, int64_t lda, cl::sycl::buffer<TYPE, 1> &x, \
              int64_t incx) {                                                                     \
        trsv(CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);           \
    }

TRSV_LAUNCHER(float, cublasStrsv)
TRSV_LAUNCHER(double, cublasDtrsv)
TRSV_LAUNCHER(std::complex<float>, cublasCtrsv)
TRSV_LAUNCHER(std::complex<double>, cublasZtrsv)

#undef TRSV_LAUNCHER

} // namespace cublas
} // namespace onemkl
