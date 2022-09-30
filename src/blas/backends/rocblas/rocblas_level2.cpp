/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Copyright (C) 2022 Heidelberg University, Engineering Mathematics and Computing Lab (EMCL) and Computing Centre (URZ)
*
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
#include "rocblas_helper.hpp"
#include "rocblas_task.hpp"

#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/blas/detail/rocblas/onemkl_blas_rocblas.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace rocblas {
namespace column_major {

// Buffer APIs

template <typename Func, typename T>
inline void gemv(Func func, sycl::queue &queue, transpose trans, int64_t m, int64_t n, T alpha,
                 sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x, int64_t incx, T beta,
                 sycl::buffer<T, 1> &y, int64_t incy) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, m, lda, incx, incy);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            auto y_ = sc.get_mem<rocDataType *>(y_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_operation(trans), m, n,
                                    (rocDataType *)&alpha, a_, lda, x_, incx, (rocDataType *)&beta,
                                    y_, incy);
        });
    });
}

#define GEMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                 \
    void gemv(sycl::queue &queue, transpose trans, int64_t m, int64_t n, TYPE alpha,         \
              sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x, int64_t incx, \
              TYPE beta, sycl::buffer<TYPE, 1> &y, int64_t incy) {                           \
        gemv(ROCBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);    \
    }

GEMV_LAUNCHER(float, rocblas_sgemv)
GEMV_LAUNCHER(double, rocblas_dgemv)
GEMV_LAUNCHER(std::complex<float>, rocblas_cgemv)
GEMV_LAUNCHER(std::complex<double>, rocblas_zgemv)
#undef GEMV_LAUNCHER

template <typename Func, typename T>
inline void gbmv(Func func, sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl,
                 int64_t ku, T alpha, sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x,
                 int64_t incx, T beta, sycl::buffer<T, 1> &y, int64_t incy) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, m, lda, kl, ku, incx, incy);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            auto y_ = sc.get_mem<rocDataType *>(y_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_operation(trans), m, n, kl, ku,
                                    (rocDataType *)&alpha, a_, lda, x_, incx, (rocDataType *)&beta,
                                    y_, incy);
        });
    });
}

#define GBMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                      \
    void gbmv(sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,  \
              TYPE alpha, sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x,        \
              int64_t incx, TYPE beta, sycl::buffer<TYPE, 1> &y, int64_t incy) {                  \
        gbmv(ROCBLAS_ROUTINE, queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); \
    }

GBMV_LAUNCHER(float, rocblas_sgbmv)
GBMV_LAUNCHER(double, rocblas_dgbmv)
GBMV_LAUNCHER(std::complex<float>, rocblas_cgbmv)
GBMV_LAUNCHER(std::complex<double>, rocblas_zgbmv)
#undef GBMV_LAUNCHER

template <typename Func, typename T>
inline void ger(Func func, sycl::queue &queue, int64_t m, int64_t n, T alpha, sycl::buffer<T, 1> &x,
                int64_t incx, sycl::buffer<T, 1> &y, int64_t incy, sycl::buffer<T, 1> &a,
                int64_t lda) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, m, lda, incx, incy);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            auto y_ = sc.get_mem<rocDataType *>(y_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, m, n, (rocDataType *)&alpha, x_, incx, y_,
                                    incy, a_, lda);
        });
    });
}

#define GER_LAUNCHER(EXT, TYPE, ROCBLAS_ROUTINE)                                                  \
    void ger##EXT(sycl::queue &queue, int64_t m, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1> &x, \
                  int64_t incx, sycl::buffer<TYPE, 1> &y, int64_t incy, sycl::buffer<TYPE, 1> &a, \
                  int64_t lda) {                                                                  \
        ger(ROCBLAS_ROUTINE, queue, m, n, alpha, x, incx, y, incy, a, lda);                       \
    }

GER_LAUNCHER(, float, rocblas_sger)
GER_LAUNCHER(, double, rocblas_dger)
GER_LAUNCHER(u, std::complex<float>, rocblas_cgeru)
GER_LAUNCHER(u, std::complex<double>, rocblas_zgeru)
GER_LAUNCHER(c, std::complex<float>, rocblas_cgerc)
GER_LAUNCHER(c, std::complex<double>, rocblas_zgerc)
#undef GER_LAUNCHER

template <typename Func, typename T>
inline void hbmv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, T alpha,
                 sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x, int64_t incx, T beta,
                 sycl::buffer<T, 1> &y, int64_t incy) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx, incy);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            auto y_ = sc.get_mem<rocDataType *>(y_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n, k,
                                    (rocDataType *)&alpha, a_, lda, x_, incx, (rocDataType *)&beta,
                                    y_, incy);
        });
    });
}

#define HBMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void hbmv(sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,           \
              sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x, int64_t incx,    \
              TYPE beta, sycl::buffer<TYPE, 1> &y, int64_t incy) {                              \
        hbmv(ROCBLAS_ROUTINE, queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy); \
    }

HBMV_LAUNCHER(std::complex<float>, rocblas_chbmv)
HBMV_LAUNCHER(std::complex<double>, rocblas_zhbmv)
#undef HBMV_LAUNCHER

template <typename Func, typename T>
inline void hemv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x, int64_t incx, T beta,
                 sycl::buffer<T, 1> &y, int64_t incy) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            auto y_ = sc.get_mem<rocDataType *>(y_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, a_, lda, x_, incx, (rocDataType *)&beta,
                                    y_, incy);
        });
    });
}

#define HEMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                 \
    void hemv(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                   \
              sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x, int64_t incx, \
              TYPE beta, sycl::buffer<TYPE, 1> &y, int64_t incy) {                           \
        hemv(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy); \
    }

HEMV_LAUNCHER(std::complex<float>, rocblas_chemv)
HEMV_LAUNCHER(std::complex<double>, rocblas_zhemv)
#undef HEMV_LAUNCHER

template <typename Func, typename ScalarType, typename DataType>
inline void her(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, ScalarType alpha,
                sycl::buffer<DataType, 1> &x, int64_t incx, sycl::buffer<DataType, 1> &a,
                int64_t lda) {
    using rocScalarType = typename RocEquivalentType<ScalarType>::Type;
    using rocDataType = typename RocEquivalentType<DataType>::Type;
    overflow_check(n, lda, incx);

    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocScalarType *)&alpha, x_, incx, a_, lda);
        });
    });
}

#define HER_LAUNCHER(SCALAR_TYPE, DATA_TYPE, ROCBLAS_ROUTINE)                            \
    void her(sycl::queue &queue, uplo upper_lower, int64_t n, SCALAR_TYPE alpha,         \
             sycl::buffer<DATA_TYPE, 1> &x, int64_t incx, sycl::buffer<DATA_TYPE, 1> &a, \
             int64_t lda) {                                                              \
        her(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, lda);             \
    }

HER_LAUNCHER(float, std::complex<float>, rocblas_cher)
HER_LAUNCHER(double, std::complex<double>, rocblas_zher)

#undef HER_LAUNCHER

template <typename Func, typename T>
inline void her2(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 sycl::buffer<T, 1> &x, int64_t incx, sycl::buffer<T, 1> &y, int64_t incy,
                 sycl::buffer<T, 1> &a, int64_t lda) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            auto y_ = sc.get_mem<rocDataType *>(y_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, x_, incx, y_, incy, a_, lda);
        });
    });
}

#define HER2_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                  \
    void her2(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                    \
              sycl::buffer<TYPE, 1> &x, int64_t incx, sycl::buffer<TYPE, 1> &y, int64_t incy, \
              sycl::buffer<TYPE, 1> &a, int64_t lda) {                                        \
        her2(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);        \
    }

HER2_LAUNCHER(std::complex<float>, rocblas_cher2)
HER2_LAUNCHER(std::complex<double>, rocblas_zher2)

#undef HER2_LAUNCHER

template <typename Func, typename T>
inline void hpmv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 sycl::buffer<T, 1> &a, sycl::buffer<T, 1> &x, int64_t incx, T beta,
                 sycl::buffer<T, 1> &y, int64_t incy) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            auto y_ = sc.get_mem<rocDataType *>(y_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, a_, x_, incx, (rocDataType *)&beta, y_,
                                    incy);
        });
    });
}

#define HPMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                               \
    void hpmv(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                 \
              sycl::buffer<TYPE, 1> &a, sycl::buffer<TYPE, 1> &x, int64_t incx, TYPE beta, \
              sycl::buffer<TYPE, 1> &y, int64_t incy) {                                    \
        hpmv(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);    \
    }

HPMV_LAUNCHER(std::complex<float>, rocblas_chpmv)
HPMV_LAUNCHER(std::complex<double>, rocblas_zhpmv)

#undef HPMV_LAUNCHER

template <typename Func, typename ScalarType, typename DataType>
inline void hpr(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, ScalarType alpha,
                sycl::buffer<DataType, 1> &x, int64_t incx, sycl::buffer<DataType, 1> &a) {
    using rocScalarType = typename RocEquivalentType<ScalarType>::Type;
    using rocDataType = typename RocEquivalentType<DataType>::Type;
    overflow_check(n, incx);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocScalarType *)&alpha, x_, incx, a_);
        });
    });
}

#define HPR_LAUNCHER(SCALAR_TYPE, DATA_TYPE, ROCBLAS_ROUTINE)                              \
    void hpr(sycl::queue &queue, uplo upper_lower, int64_t n, SCALAR_TYPE alpha,           \
             sycl::buffer<DATA_TYPE, 1> &x, int64_t incx, sycl::buffer<DATA_TYPE, 1> &a) { \
        hpr(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a);                    \
    }

HPR_LAUNCHER(float, std::complex<float>, rocblas_chpr)
HPR_LAUNCHER(double, std::complex<double>, rocblas_zhpr)

#undef HPR_LAUNCHER

template <typename Func, typename T>
inline void hpr2(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 sycl::buffer<T, 1> &x, int64_t incx, sycl::buffer<T, 1> &y, int64_t incy,
                 sycl::buffer<T, 1> &a) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            auto y_ = sc.get_mem<rocDataType *>(y_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, x_, incx, y_, incy, a_);
        });
    });
}

#define HPR2_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                  \
    void hpr2(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                    \
              sycl::buffer<TYPE, 1> &x, int64_t incx, sycl::buffer<TYPE, 1> &y, int64_t incy, \
              sycl::buffer<TYPE, 1> &a) {                                                     \
        hpr2(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a);             \
    }

HPR2_LAUNCHER(std::complex<float>, rocblas_chpr2)
HPR2_LAUNCHER(std::complex<double>, rocblas_zhpr2)

#undef HPR2_LAUNCHER

template <typename Func, typename T>
inline void sbmv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, T alpha,
                 sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x, int64_t incx, T beta,
                 sycl::buffer<T, 1> &y, int64_t incy) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx, incy);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            auto y_ = sc.get_mem<rocDataType *>(y_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n, k,
                                    (rocDataType *)&alpha, a_, lda, x_, incx, (rocDataType *)&beta,
                                    y_, incy);
        });
    });
}

#define SBMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void sbmv(sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,           \
              sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x, int64_t incx,    \
              TYPE beta, sycl::buffer<TYPE, 1> &y, int64_t incy) {                              \
        sbmv(ROCBLAS_ROUTINE, queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy); \
    }

SBMV_LAUNCHER(float, rocblas_ssbmv)
SBMV_LAUNCHER(double, rocblas_dsbmv)

#undef SBMV_LAUNCHER

template <typename Func, typename T>
inline void symv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x, int64_t incx, T beta,
                 sycl::buffer<T, 1> &y, int64_t incy) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            auto y_ = sc.get_mem<rocDataType *>(y_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, a_, lda, x_, incx, (rocDataType *)&beta,
                                    y_, incy);
        });
    });
}

#define SYMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                 \
    void symv(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                   \
              sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x, int64_t incx, \
              TYPE beta, sycl::buffer<TYPE, 1> &y, int64_t incy) {                           \
        symv(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy); \
    }

SYMV_LAUNCHER(float, rocblas_ssymv)
SYMV_LAUNCHER(double, rocblas_dsymv)

#undef SYMV_LAUNCHER

template <typename Func, typename T>
inline void syr(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                sycl::buffer<T, 1> &x, int64_t incx, sycl::buffer<T, 1> &a, int64_t lda) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, lda, incx);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, x_, incx, a_, lda);
        });
    });
}

#define SYR_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                   \
    void syr(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                     \
             sycl::buffer<TYPE, 1> &x, int64_t incx, sycl::buffer<TYPE, 1> &a, int64_t lda) { \
        syr(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, lda);                  \
    }

SYR_LAUNCHER(float, rocblas_ssyr)
SYR_LAUNCHER(double, rocblas_dsyr)
// Intel does not support the following two
SYR_LAUNCHER(std::complex<float>, rocblas_csyr)
SYR_LAUNCHER(std::complex<double>, rocblas_zsyr)
#undef SYR_LAUNCHER

template <typename Func, typename T>
inline void syr2(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 sycl::buffer<T, 1> &x, int64_t incx, sycl::buffer<T, 1> &y, int64_t incy,
                 sycl::buffer<T, 1> &a, int64_t lda) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            auto y_ = sc.get_mem<rocDataType *>(y_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, x_, incx, y_, incy, a_, lda);
        });
    });
}

#define SYR2_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                  \
    void syr2(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                    \
              sycl::buffer<TYPE, 1> &x, int64_t incx, sycl::buffer<TYPE, 1> &y, int64_t incy, \
              sycl::buffer<TYPE, 1> &a, int64_t lda) {                                        \
        syr2(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);        \
    }

SYR2_LAUNCHER(float, rocblas_ssyr2)
SYR2_LAUNCHER(double, rocblas_dsyr2)
// Intel does not support the following two
SYR2_LAUNCHER(std::complex<float>, rocblas_csyr2)
SYR2_LAUNCHER(std::complex<double>, rocblas_zsyr2)

#undef SYR2_LAUNCHER

template <typename Func, typename T>
inline void spmv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 sycl::buffer<T, 1> &a, sycl::buffer<T, 1> &x, int64_t incx, T beta,
                 sycl::buffer<T, 1> &y, int64_t incy) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            auto y_ = sc.get_mem<rocDataType *>(y_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, a_, x_, incx, (rocDataType *)&beta, y_,
                                    incy);
        });
    });
}

#define SPMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                               \
    void spmv(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                 \
              sycl::buffer<TYPE, 1> &a, sycl::buffer<TYPE, 1> &x, int64_t incx, TYPE beta, \
              sycl::buffer<TYPE, 1> &y, int64_t incy) {                                    \
        spmv(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);    \
    }

SPMV_LAUNCHER(float, rocblas_sspmv)
SPMV_LAUNCHER(double, rocblas_dspmv)

#undef SPMV_LAUNCHER

template <typename Func, typename T>
inline void spr(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                sycl::buffer<T, 1> &x, int64_t incx, sycl::buffer<T, 1> &a) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, x_, incx, a_);
        });
    });
}

#define SPR_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                      \
    void spr(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,        \
             sycl::buffer<TYPE, 1> &x, int64_t incx, sycl::buffer<TYPE, 1> &a) { \
        spr(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a);          \
    }

SPR_LAUNCHER(float, rocblas_sspr)
SPR_LAUNCHER(double, rocblas_dspr)

#undef SPR_LAUNCHER

template <typename Func, typename T>
inline void spr2(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 sycl::buffer<T, 1> &x, int64_t incx, sycl::buffer<T, 1> &y, int64_t incy,
                 sycl::buffer<T, 1> &a) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            auto y_ = sc.get_mem<rocDataType *>(y_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, x_, incx, y_, incy, a_);
        });
    });
}

#define SPR2_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                  \
    void spr2(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                    \
              sycl::buffer<TYPE, 1> &x, int64_t incx, sycl::buffer<TYPE, 1> &y, int64_t incy, \
              sycl::buffer<TYPE, 1> &a) {                                                     \
        spr2(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a);             \
    }

SPR2_LAUNCHER(float, rocblas_sspr2)
SPR2_LAUNCHER(double, rocblas_dspr2)

#undef SPR2_LAUNCHER

template <typename Func, typename T>
inline void tbmv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                 int64_t n, int64_t k, sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x,
                 int64_t incx) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                    get_rocblas_operation(trans), get_rocblas_diag_type(unit_diag),
                                    n, k, a_, lda, x_, incx);
        });
    });
}

#define TBMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void tbmv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              int64_t k, sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x,       \
              int64_t incx) {                                                                   \
        tbmv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);     \
    }

TBMV_LAUNCHER(float, rocblas_stbmv)
TBMV_LAUNCHER(double, rocblas_dtbmv)
TBMV_LAUNCHER(std::complex<float>, rocblas_ctbmv)
TBMV_LAUNCHER(std::complex<double>, rocblas_ztbmv)

#undef TBMV_LAUNCHER

template <typename Func, typename T>
inline void tbsv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                 int64_t n, int64_t k, sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x,
                 int64_t incx) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                    get_rocblas_operation(trans), get_rocblas_diag_type(unit_diag),
                                    n, k, a_, lda, x_, incx);
        });
    });
}

#define TBSV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void tbsv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              int64_t k, sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x,       \
              int64_t incx) {                                                                   \
        tbsv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);     \
    }

TBSV_LAUNCHER(float, rocblas_stbsv)
TBSV_LAUNCHER(double, rocblas_dtbsv)
TBSV_LAUNCHER(std::complex<float>, rocblas_ctbsv)
TBSV_LAUNCHER(std::complex<double>, rocblas_ztbsv)

#undef TBSV_LAUNCHER

template <typename Func, typename T>
inline void tpmv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                 int64_t n, sycl::buffer<T, 1> &a, sycl::buffer<T, 1> &x, int64_t incx) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                    get_rocblas_operation(trans), get_rocblas_diag_type(unit_diag),
                                    n, a_, x_, incx);
        });
    });
}

#define TPMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void tpmv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              sycl::buffer<TYPE, 1> &a, sycl::buffer<TYPE, 1> &x, int64_t incx) {               \
        tpmv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, x, incx);             \
    }

TPMV_LAUNCHER(float, rocblas_stpmv)
TPMV_LAUNCHER(double, rocblas_dtpmv)
TPMV_LAUNCHER(std::complex<float>, rocblas_ctpmv)
TPMV_LAUNCHER(std::complex<double>, rocblas_ztpmv)

#undef TPMV_LAUNCHER

template <typename Func, typename T>
inline void tpsv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                 int64_t n, sycl::buffer<T, 1> &a, sycl::buffer<T, 1> &x, int64_t incx) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                    get_rocblas_operation(trans), get_rocblas_diag_type(unit_diag),
                                    n, a_, x_, incx);
        });
    });
}

#define TPSV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void tpsv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              sycl::buffer<TYPE, 1> &a, sycl::buffer<TYPE, 1> &x, int64_t incx) {               \
        tpsv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, x, incx);             \
    }

TPSV_LAUNCHER(float, rocblas_stpsv)
TPSV_LAUNCHER(double, rocblas_dtpsv)
TPSV_LAUNCHER(std::complex<float>, rocblas_ctpsv)
TPSV_LAUNCHER(std::complex<double>, rocblas_ztpsv)

#undef TPSV_LAUNCHER

template <typename Func, typename T>
inline void trmv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                 int64_t n, sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x,
                 int64_t incx) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, lda, incx);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                    get_rocblas_operation(trans), get_rocblas_diag_type(unit_diag),
                                    n, a_, lda, x_, incx);
        });
    });
}

#define TRMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void trmv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x, int64_t incx) {  \
        trmv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);        \
    }

TRMV_LAUNCHER(float, rocblas_strmv)
TRMV_LAUNCHER(double, rocblas_dtrmv)
TRMV_LAUNCHER(std::complex<float>, rocblas_ctrmv)
TRMV_LAUNCHER(std::complex<double>, rocblas_ztrmv)

#undef TRMV_LAUNCHER

template <typename Func, typename T>
inline void trsv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                 int64_t n, sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x,
                 int64_t incx) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, lda, incx);
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType *>(a_acc);
            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                    get_rocblas_operation(trans), get_rocblas_diag_type(unit_diag),
                                    n, a_, lda, x_, incx);
        });
    });
}

#define TRSV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void trsv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x, int64_t incx) {  \
        trsv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);        \
    }

TRSV_LAUNCHER(float, rocblas_strsv)
TRSV_LAUNCHER(double, rocblas_dtrsv)
TRSV_LAUNCHER(std::complex<float>, rocblas_ctrsv)
TRSV_LAUNCHER(std::complex<double>, rocblas_ztrsv)

#undef TRSV_LAUNCHER

// USM APIs

template <typename Func, typename T>
inline sycl::event gemv(Func func, sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                        T alpha, const T *a, int64_t lda, const T *x, int64_t incx, T beta, T *y,
                        int64_t incy, const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, m, lda, incx, incy);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto x_ = reinterpret_cast<const rocDataType *>(x);
            auto y_ = reinterpret_cast<rocDataType *>(y);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_operation(trans), m, n,
                                    (rocDataType *)&alpha, a_, lda, x_, incx, (rocDataType *)&beta,
                                    y_, incy);
        });
    });
    return done;
}

#define GEMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event gemv(sycl::queue &queue, transpose trans, int64_t m, int64_t n, TYPE alpha,       \
                     const TYPE *a, int64_t lda, const TYPE *x, int64_t incx, TYPE beta, TYPE *y, \
                     int64_t incy, const std::vector<sycl::event> &dependencies) {                \
        return gemv(ROCBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy,   \
                    dependencies);                                                                \
    }

GEMV_LAUNCHER_USM(float, rocblas_sgemv)
GEMV_LAUNCHER_USM(double, rocblas_dgemv)
GEMV_LAUNCHER_USM(std::complex<float>, rocblas_cgemv)
GEMV_LAUNCHER_USM(std::complex<double>, rocblas_zgemv)
#undef GEMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event gbmv(Func func, sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                        int64_t kl, int64_t ku, T alpha, const T *a, int64_t lda, const T *x,
                        int64_t incx, T beta, T *y, int64_t incy,
                        const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, m, lda, kl, ku, incx, incy);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto x_ = reinterpret_cast<const rocDataType *>(x);
            auto y_ = reinterpret_cast<rocDataType *>(y);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_operation(trans), m, n, kl, ku,
                                    (rocDataType *)&alpha, a_, lda, x_, incx, (rocDataType *)&beta,
                                    y_, incy);
        });
    });
    return done;
}

#define GBMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event gbmv(sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl,       \
                     int64_t ku, TYPE alpha, const TYPE *a, int64_t lda, const TYPE *x,           \
                     int64_t incx, TYPE beta, TYPE *y, int64_t incy,                              \
                     const std::vector<sycl::event> &dependencies) {                              \
        return gbmv(ROCBLAS_ROUTINE, queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, \
                    incy, dependencies);                                                          \
    }

GBMV_LAUNCHER_USM(float, rocblas_sgbmv)
GBMV_LAUNCHER_USM(double, rocblas_dgbmv)
GBMV_LAUNCHER_USM(std::complex<float>, rocblas_cgbmv)
GBMV_LAUNCHER_USM(std::complex<double>, rocblas_zgbmv)
#undef GBMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event ger(Func func, sycl::queue &queue, int64_t m, int64_t n, T alpha, const T *x,
                       int64_t incx, const T *y, int64_t incy, T *a, int64_t lda,
                       const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, m, lda, incx, incy);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<rocDataType *>(a);
            auto x_ = reinterpret_cast<const rocDataType *>(x);
            auto y_ = reinterpret_cast<const rocDataType *>(y);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, m, n, (rocDataType *)&alpha, x_, incx, y_,
                                    incy, a_, lda);
        });
    });
    return done;
}

#define GER_LAUNCHER_USM(EXT, TYPE, ROCBLAS_ROUTINE)                                             \
    sycl::event ger##EXT(sycl::queue &queue, int64_t m, int64_t n, TYPE alpha, const TYPE *x,    \
                         int64_t incx, const TYPE *y, int64_t incy, TYPE *a, int64_t lda,        \
                         const std::vector<sycl::event> &dependencies) {                         \
        return ger(ROCBLAS_ROUTINE, queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies); \
    }

GER_LAUNCHER_USM(, float, rocblas_sger)
GER_LAUNCHER_USM(, double, rocblas_dger)
GER_LAUNCHER_USM(u, std::complex<float>, rocblas_cgeru)
GER_LAUNCHER_USM(u, std::complex<double>, rocblas_zgeru)
GER_LAUNCHER_USM(c, std::complex<float>, rocblas_cgerc)
GER_LAUNCHER_USM(c, std::complex<double>, rocblas_zgerc)
#undef GER_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event hbmv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k,
                        T alpha, const T *a, int64_t lda, const T *x, int64_t incx, T beta, T *y,
                        int64_t incy, const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx, incy);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto x_ = reinterpret_cast<const rocDataType *>(x);
            auto y_ = reinterpret_cast<rocDataType *>(y);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n, k,
                                    (rocDataType *)&alpha, a_, lda, x_, incx, (rocDataType *)&beta,
                                    y_, incy);
        });
    });
    return done;
}

#define HBMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event hbmv(sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,      \
                     const TYPE *a, int64_t lda, const TYPE *x, int64_t incx, TYPE beta, TYPE *y, \
                     int64_t incy, const std::vector<sycl::event> &dependencies) {                \
        return hbmv(ROCBLAS_ROUTINE, queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,   \
                    incy, dependencies);                                                          \
    }

HBMV_LAUNCHER_USM(std::complex<float>, rocblas_chbmv)
HBMV_LAUNCHER_USM(std::complex<double>, rocblas_zhbmv)
#undef HBMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event hemv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                        const T *a, int64_t lda, const T *x, int64_t incx, T beta, T *y,
                        int64_t incy, const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto x_ = reinterpret_cast<const rocDataType *>(x);
            auto y_ = reinterpret_cast<rocDataType *>(y);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, a_, lda, x_, incx, (rocDataType *)&beta,
                                    y_, incy);
        });
    });
    return done;
}

#define HEMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                   \
    sycl::event hemv(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *a,   \
                     int64_t lda, const TYPE *x, int64_t incx, TYPE beta, TYPE *y, int64_t incy,   \
                     const std::vector<sycl::event> &dependencies) {                               \
        return hemv(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, \
                    dependencies);                                                                 \
    }

HEMV_LAUNCHER_USM(std::complex<float>, rocblas_chemv)
HEMV_LAUNCHER_USM(std::complex<double>, rocblas_zhemv)
#undef HEMV_LAUNCHER_USM

template <typename Func, typename ScalarType, typename DataType>
inline sycl::event her(Func func, sycl::queue &queue, uplo upper_lower, int64_t n,
                       const ScalarType alpha, const DataType *x, int64_t incx, DataType *a,
                       int64_t lda, const std::vector<sycl::event> &dependencies) {
    using rocScalarType = typename RocEquivalentType<ScalarType>::Type;
    using rocDataType = typename RocEquivalentType<DataType>::Type;
    overflow_check(n, lda, incx);

    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<rocDataType *>(a);
            auto x_ = reinterpret_cast<const rocDataType *>(x);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocScalarType *)&alpha, x_, incx, a_, lda);
        });
    });
    return done;
}

#define HER_LAUNCHER_USM(SCALAR_TYPE, DATA_TYPE, ROCBLAS_ROUTINE)                                 \
    sycl::event her(sycl::queue &queue, uplo upper_lower, int64_t n, const SCALAR_TYPE alpha,     \
                    const DATA_TYPE *x, int64_t incx, DATA_TYPE *a, int64_t lda,                  \
                    const std::vector<sycl::event> &dependencies) {                               \
        return her(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, lda, dependencies); \
    }

HER_LAUNCHER_USM(float, std::complex<float>, rocblas_cher)
HER_LAUNCHER_USM(double, std::complex<double>, rocblas_zher)

#undef HER_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event her2(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                        const T *x, int64_t incx, const T *y, int64_t incy, T *a, int64_t lda,
                        const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<rocDataType *>(a);
            auto x_ = reinterpret_cast<const rocDataType *>(x);
            auto y_ = reinterpret_cast<const rocDataType *>(y);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, x_, incx, y_, incy, a_, lda);
        });
    });
    return done;
}

#define HER2_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                 \
    sycl::event her2(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *x, \
                     int64_t incx, const TYPE *y, int64_t incy, TYPE *a, int64_t lda,            \
                     const std::vector<sycl::event> &dependencies) {                             \
        return her2(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a, lda,     \
                    dependencies);                                                               \
    }

HER2_LAUNCHER_USM(std::complex<float>, rocblas_cher2)
HER2_LAUNCHER_USM(std::complex<double>, rocblas_zher2)

#undef HER2_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event hpmv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                        const T *a, const T *x, int64_t incx, T beta, T *y, int64_t incy,
                        const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto x_ = reinterpret_cast<const rocDataType *>(x);
            auto y_ = reinterpret_cast<rocDataType *>(y);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, a_, x_, incx, (rocDataType *)&beta, y_,
                                    incy);
        });
    });
    return done;
}

#define HPMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                 \
    sycl::event hpmv(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *a, \
                     const TYPE *x, int64_t incx, TYPE beta, TYPE *y, int64_t incy,              \
                     const std::vector<sycl::event> &dependencies) {                             \
        return hpmv(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, a, x, incx, beta, y, incy,    \
                    dependencies);                                                               \
    }

HPMV_LAUNCHER_USM(std::complex<float>, rocblas_chpmv)
HPMV_LAUNCHER_USM(std::complex<double>, rocblas_zhpmv)

#undef HPMV_LAUNCHER_USM

template <typename Func, typename ScalarType, typename DataType>
inline sycl::event hpr(Func func, sycl::queue &queue, uplo upper_lower, int64_t n,
                       const ScalarType alpha, const DataType *x, int64_t incx, DataType *a,
                       const std::vector<sycl::event> &dependencies) {
    using rocScalarType = typename RocEquivalentType<ScalarType>::Type;
    using rocDataType = typename RocEquivalentType<DataType>::Type;
    overflow_check(n, incx);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<rocDataType *>(a);
            auto x_ = reinterpret_cast<const rocDataType *>(x);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocScalarType *)&alpha, x_, incx, a_);
        });
    });
    return done;
}

#define HPR_LAUNCHER_USM(SCALAR_TYPE, DATA_TYPE, ROCBLAS_ROUTINE)                             \
    sycl::event hpr(sycl::queue &queue, uplo upper_lower, int64_t n, const SCALAR_TYPE alpha, \
                    const DATA_TYPE *x, int64_t incx, DATA_TYPE *a,                           \
                    const std::vector<sycl::event> &dependencies) {                           \
        return hpr(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, dependencies);  \
    }

HPR_LAUNCHER_USM(float, std::complex<float>, rocblas_chpr)
HPR_LAUNCHER_USM(double, std::complex<double>, rocblas_zhpr)

#undef HPR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event hpr2(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                        const T *x, int64_t incx, const T *y, int64_t incy, T *a,
                        const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<rocDataType *>(a);
            auto x_ = reinterpret_cast<const rocDataType *>(x);
            auto y_ = reinterpret_cast<const rocDataType *>(y);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, x_, incx, y_, incy, a_);
        });
    });
    return done;
}

#define HPR2_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                 \
    sycl::event hpr2(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *x, \
                     int64_t incx, const TYPE *y, int64_t incy, TYPE *a,                         \
                     const std::vector<sycl::event> &dependencies) {                             \
        return hpr2(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a,          \
                    dependencies);                                                               \
    }

HPR2_LAUNCHER_USM(std::complex<float>, rocblas_chpr2)
HPR2_LAUNCHER_USM(std::complex<double>, rocblas_zhpr2)

#undef HPR2_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event sbmv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k,
                        T alpha, const T *a, int64_t lda, const T *x, int64_t incx, T beta, T *y,
                        int64_t incy, const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx, incy);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto x_ = reinterpret_cast<const rocDataType *>(x);
            auto y_ = reinterpret_cast<rocDataType *>(y);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n, k,
                                    (rocDataType *)&alpha, a_, lda, x_, incx, (rocDataType *)&beta,
                                    y_, incy);
        });
    });
    return done;
}

#define SBMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event sbmv(sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,      \
                     const TYPE *a, int64_t lda, const TYPE *x, int64_t incx, TYPE beta, TYPE *y, \
                     int64_t incy, const std::vector<sycl::event> &dependencies) {                \
        return sbmv(ROCBLAS_ROUTINE, queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,   \
                    incy, dependencies);                                                          \
    }

SBMV_LAUNCHER_USM(float, rocblas_ssbmv)
SBMV_LAUNCHER_USM(double, rocblas_dsbmv)

#undef SBMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event symv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                        const T *a, int64_t lda, const T *x, int64_t incx, T beta, T *y,
                        int64_t incy, const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto x_ = reinterpret_cast<const rocDataType *>(x);
            auto y_ = reinterpret_cast<rocDataType *>(y);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, a_, lda, x_, incx, (rocDataType *)&beta,
                                    y_, incy);
        });
    });
    return done;
}

#define SYMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                   \
    sycl::event symv(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *a,   \
                     int64_t lda, const TYPE *x, int64_t incx, TYPE beta, TYPE *y, int64_t incy,   \
                     const std::vector<sycl::event> &dependencies) {                               \
        return symv(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, \
                    dependencies);                                                                 \
    }

SYMV_LAUNCHER_USM(float, rocblas_ssymv)
SYMV_LAUNCHER_USM(double, rocblas_dsymv)

#undef SYMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event syr(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                       const T *x, int64_t incx, T *a, int64_t lda,
                       const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, lda, incx);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<rocDataType *>(a);
            auto x_ = reinterpret_cast<const rocDataType *>(x);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, x_, incx, a_, lda);
        });
    });
    return done;
}

#define SYR_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                   \
    sycl::event syr(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *x,   \
                    int64_t incx, TYPE *a, int64_t lda,                                           \
                    const std::vector<sycl::event> &dependencies) {                               \
        return syr(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, lda, dependencies); \
    }

SYR_LAUNCHER_USM(float, rocblas_ssyr)
SYR_LAUNCHER_USM(double, rocblas_dsyr)
// Intel does not support the following two
SYR_LAUNCHER_USM(std::complex<float>, rocblas_csyr)
SYR_LAUNCHER_USM(std::complex<double>, rocblas_zsyr)
#undef SYR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event syr2(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                        const T *x, int64_t incx, const T *y, int64_t incy, T *a, int64_t lda,
                        const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<rocDataType *>(a);
            auto x_ = reinterpret_cast<const rocDataType *>(x);
            auto y_ = reinterpret_cast<const rocDataType *>(y);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, x_, incx, y_, incy, a_, lda);
        });
    });
    return done;
}

#define SYR2_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                 \
    sycl::event syr2(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *x, \
                     int64_t incx, const TYPE *y, int64_t incy, TYPE *a, int64_t lda,            \
                     const std::vector<sycl::event> &dependencies) {                             \
        return syr2(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a, lda,     \
                    dependencies);                                                               \
    }

SYR2_LAUNCHER_USM(float, rocblas_ssyr2)
SYR2_LAUNCHER_USM(double, rocblas_dsyr2)
// Intel does not support the following two
SYR2_LAUNCHER_USM(std::complex<float>, rocblas_csyr2)
SYR2_LAUNCHER_USM(std::complex<double>, rocblas_zsyr2)

#undef SYR2_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event spmv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                        const T *a, const T *x, int64_t incx, T beta, T *y, int64_t incy,
                        const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto x_ = reinterpret_cast<const rocDataType *>(x);
            auto y_ = reinterpret_cast<rocDataType *>(y);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, a_, x_, incx, (rocDataType *)&beta, y_,
                                    incy);
        });
    });
    return done;
}

#define SPMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                 \
    sycl::event spmv(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *a, \
                     const TYPE *x, int64_t incx, TYPE beta, TYPE *y, int64_t incy,              \
                     const std::vector<sycl::event> &dependencies) {                             \
        return spmv(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, a, x, incx, beta, y, incy,    \
                    dependencies);                                                               \
    }

SPMV_LAUNCHER_USM(float, rocblas_sspmv)
SPMV_LAUNCHER_USM(double, rocblas_dspmv)

#undef SPMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event spr(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                       const T *x, int64_t incx, T *a,
                       const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<rocDataType *>(a);
            auto x_ = reinterpret_cast<const rocDataType *>(x);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, x_, incx, a_);
        });
    });
    return done;
}

#define SPR_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                 \
    sycl::event spr(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *x, \
                    int64_t incx, TYPE *a, const std::vector<sycl::event> &dependencies) {      \
        return spr(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, dependencies);    \
    }

SPR_LAUNCHER_USM(float, rocblas_sspr)
SPR_LAUNCHER_USM(double, rocblas_dspr)

#undef SPR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event spr2(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                        const T *x, int64_t incx, const T *y, int64_t incy, T *a,
                        const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<rocDataType *>(a);
            auto x_ = reinterpret_cast<const rocDataType *>(x);
            auto y_ = reinterpret_cast<const rocDataType *>(y);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower), n,
                                    (rocDataType *)&alpha, x_, incx, y_, incy, a_);
        });
    });
    return done;
}

#define SPR2_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                 \
    sycl::event spr2(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *x, \
                     int64_t incx, const TYPE *y, int64_t incy, TYPE *a,                         \
                     const std::vector<sycl::event> &dependencies) {                             \
        return spr2(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a,          \
                    dependencies);                                                               \
    }

SPR2_LAUNCHER_USM(float, rocblas_sspr2)
SPR2_LAUNCHER_USM(double, rocblas_dspr2)

#undef SPR2_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event tbmv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans,
                        diag unit_diag, int64_t n, int64_t k, const T *a, int64_t lda, T *x,
                        int64_t incx, const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto x_ = reinterpret_cast<rocDataType *>(x);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                    get_rocblas_operation(trans), get_rocblas_diag_type(unit_diag),
                                    n, k, a_, lda, x_, incx);
        });
    });
    return done;
}

#define TBMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event tbmv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,       \
                     int64_t n, int64_t k, const TYPE *a, int64_t lda, TYPE *x, int64_t incx,     \
                     const std::vector<sycl::event> &dependencies) {                              \
        return tbmv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, \
                    dependencies);                                                                \
    }

TBMV_LAUNCHER_USM(float, rocblas_stbmv)
TBMV_LAUNCHER_USM(double, rocblas_dtbmv)
TBMV_LAUNCHER_USM(std::complex<float>, rocblas_ctbmv)
TBMV_LAUNCHER_USM(std::complex<double>, rocblas_ztbmv)

#undef TBMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event tbsv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans,
                        diag unit_diag, int64_t n, int64_t k, const T *a, int64_t lda, T *x,
                        int64_t incx, const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto x_ = reinterpret_cast<rocDataType *>(x);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                    get_rocblas_operation(trans), get_rocblas_diag_type(unit_diag),
                                    n, k, a_, lda, x_, incx);
        });
    });
    return done;
}

#define TBSV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event tbsv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,       \
                     int64_t n, int64_t k, const TYPE *a, int64_t lda, TYPE *x, int64_t incx,     \
                     const std::vector<sycl::event> &dependencies) {                              \
        return tbsv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, \
                    dependencies);                                                                \
    }

TBSV_LAUNCHER_USM(float, rocblas_stbsv)
TBSV_LAUNCHER_USM(double, rocblas_dtbsv)
TBSV_LAUNCHER_USM(std::complex<float>, rocblas_ctbsv)
TBSV_LAUNCHER_USM(std::complex<double>, rocblas_ztbsv)

#undef TBSV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event tpmv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans,
                        diag unit_diag, int64_t n, const T *a, T *x, int64_t incx,
                        const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto x_ = reinterpret_cast<rocDataType *>(x);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                    get_rocblas_operation(trans), get_rocblas_diag_type(unit_diag),
                                    n, a_, x_, incx);
        });
    });
    return done;
}

#define TPMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                            \
    sycl::event tpmv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, \
                     int64_t n, const TYPE *a, TYPE *x, int64_t incx,                       \
                     const std::vector<sycl::event> &dependencies) {                        \
        return tpmv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, x, incx,   \
                    dependencies);                                                          \
    }

TPMV_LAUNCHER_USM(float, rocblas_stpmv)
TPMV_LAUNCHER_USM(double, rocblas_dtpmv)
TPMV_LAUNCHER_USM(std::complex<float>, rocblas_ctpmv)
TPMV_LAUNCHER_USM(std::complex<double>, rocblas_ztpmv)

#undef TPMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event tpsv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans,
                        diag unit_diag, int64_t n, const T *a, T *x, int64_t incx,
                        const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto x_ = reinterpret_cast<rocDataType *>(x);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                    get_rocblas_operation(trans), get_rocblas_diag_type(unit_diag),
                                    n, a_, x_, incx);
        });
    });
    return done;
}

#define TPSV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                            \
    sycl::event tpsv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, \
                     int64_t n, const TYPE *a, TYPE *x, int64_t incx,                       \
                     const std::vector<sycl::event> &dependencies) {                        \
        return tpsv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, x, incx,   \
                    dependencies);                                                          \
    }

TPSV_LAUNCHER_USM(float, rocblas_stpsv)
TPSV_LAUNCHER_USM(double, rocblas_dtpsv)
TPSV_LAUNCHER_USM(std::complex<float>, rocblas_ctpsv)
TPSV_LAUNCHER_USM(std::complex<double>, rocblas_ztpsv)

#undef TPSV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event trmv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans,
                        diag unit_diag, int64_t n, const T *a, int64_t lda, T *x, int64_t incx,
                        const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, lda, incx);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto x_ = reinterpret_cast<rocDataType *>(x);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                    get_rocblas_operation(trans), get_rocblas_diag_type(unit_diag),
                                    n, a_, lda, x_, incx);
        });
    });
    return done;
}

#define TRMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                               \
    sycl::event trmv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,    \
                     int64_t n, const TYPE *a, int64_t lda, TYPE *x, int64_t incx,             \
                     const std::vector<sycl::event> &dependencies) {                           \
        return trmv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, \
                    dependencies);                                                             \
    }

TRMV_LAUNCHER_USM(float, rocblas_strmv)
TRMV_LAUNCHER_USM(double, rocblas_dtrmv)
TRMV_LAUNCHER_USM(std::complex<float>, rocblas_ctrmv)
TRMV_LAUNCHER_USM(std::complex<double>, rocblas_ztrmv)

#undef TRMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event trsv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans,
                        diag unit_diag, int64_t n, const T *a, int64_t lda, T *x, int64_t incx,
                        const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, lda, incx);
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto x_ = reinterpret_cast<rocDataType *>(x);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                    get_rocblas_operation(trans), get_rocblas_diag_type(unit_diag),
                                    n, a_, lda, x_, incx);
        });
    });
    return done;
}

#define TRSV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                               \
    sycl::event trsv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,    \
                     int64_t n, const TYPE *a, int64_t lda, TYPE *x, int64_t incx,             \
                     const std::vector<sycl::event> &dependencies) {                           \
        return trsv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, \
                    dependencies);                                                             \
    }

TRSV_LAUNCHER_USM(float, rocblas_strsv)
TRSV_LAUNCHER_USM(double, rocblas_dtrsv)
TRSV_LAUNCHER_USM(std::complex<float>, rocblas_ctrsv)
TRSV_LAUNCHER_USM(std::complex<double>, rocblas_ztrsv)

#undef TRSV_LAUNCHER_USM

} // namespace column_major
namespace row_major {

// Buffer APIs

template <typename Func, typename T>
inline void gemv(Func func, sycl::queue &queue, transpose trans, int64_t m, int64_t n, T alpha,
                 sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x, int64_t incx, T beta,
                 sycl::buffer<T, 1> &y, int64_t incy) {
    throw unimplemented("blas", "gemv", "for row_major layout");
}

#define GEMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                 \
    void gemv(sycl::queue &queue, transpose trans, int64_t m, int64_t n, TYPE alpha,         \
              sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x, int64_t incx, \
              TYPE beta, sycl::buffer<TYPE, 1> &y, int64_t incy) {                           \
        gemv(ROCBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);    \
    }

GEMV_LAUNCHER(float, rocblas_sgemv)
GEMV_LAUNCHER(double, rocblas_dgemv)
GEMV_LAUNCHER(std::complex<float>, rocblas_cgemv)
GEMV_LAUNCHER(std::complex<double>, rocblas_zgemv)
#undef GEMV_LAUNCHER

template <typename Func, typename T>
inline void gbmv(Func func, sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl,
                 int64_t ku, T alpha, sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x,
                 int64_t incx, T beta, sycl::buffer<T, 1> &y, int64_t incy) {
    throw unimplemented("blas", "gbmv", "for row_major layout");
}

#define GBMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                      \
    void gbmv(sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,  \
              TYPE alpha, sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x,        \
              int64_t incx, TYPE beta, sycl::buffer<TYPE, 1> &y, int64_t incy) {                  \
        gbmv(ROCBLAS_ROUTINE, queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); \
    }

GBMV_LAUNCHER(float, rocblas_sgbmv)
GBMV_LAUNCHER(double, rocblas_dgbmv)
GBMV_LAUNCHER(std::complex<float>, rocblas_cgbmv)
GBMV_LAUNCHER(std::complex<double>, rocblas_zgbmv)
#undef GBMV_LAUNCHER

template <typename Func, typename T>
inline void ger(Func func, sycl::queue &queue, int64_t m, int64_t n, T alpha, sycl::buffer<T, 1> &x,
                int64_t incx, sycl::buffer<T, 1> &y, int64_t incy, sycl::buffer<T, 1> &a,
                int64_t lda) {
    throw unimplemented("blas", "ger", "for row_major layout");
}

#define GER_LAUNCHER(EXT, TYPE, ROCBLAS_ROUTINE)                                                  \
    void ger##EXT(sycl::queue &queue, int64_t m, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1> &x, \
                  int64_t incx, sycl::buffer<TYPE, 1> &y, int64_t incy, sycl::buffer<TYPE, 1> &a, \
                  int64_t lda) {                                                                  \
        ger(ROCBLAS_ROUTINE, queue, m, n, alpha, x, incx, y, incy, a, lda);                       \
    }

GER_LAUNCHER(, float, rocblas_sger)
GER_LAUNCHER(, double, rocblas_dger)
GER_LAUNCHER(u, std::complex<float>, rocblas_cgeru)
GER_LAUNCHER(u, std::complex<double>, rocblas_zgeru)
GER_LAUNCHER(c, std::complex<float>, rocblas_cgerc)
GER_LAUNCHER(c, std::complex<double>, rocblas_zgerc)
#undef GER_LAUNCHER

template <typename Func, typename T>
inline void hbmv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, T alpha,
                 sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x, int64_t incx, T beta,
                 sycl::buffer<T, 1> &y, int64_t incy) {
    throw unimplemented("blas", "hbmv", "for row_major layout");
}

#define HBMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void hbmv(sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,           \
              sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x, int64_t incx,    \
              TYPE beta, sycl::buffer<TYPE, 1> &y, int64_t incy) {                              \
        hbmv(ROCBLAS_ROUTINE, queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy); \
    }

HBMV_LAUNCHER(std::complex<float>, rocblas_chbmv)
HBMV_LAUNCHER(std::complex<double>, rocblas_zhbmv)
#undef HBMV_LAUNCHER

template <typename Func, typename T>
inline void hemv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x, int64_t incx, T beta,
                 sycl::buffer<T, 1> &y, int64_t incy) {
    throw unimplemented("blas", "hemv", "for row_major layout");
}

#define HEMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                 \
    void hemv(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                   \
              sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x, int64_t incx, \
              TYPE beta, sycl::buffer<TYPE, 1> &y, int64_t incy) {                           \
        hemv(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy); \
    }

HEMV_LAUNCHER(std::complex<float>, rocblas_chemv)
HEMV_LAUNCHER(std::complex<double>, rocblas_zhemv)
#undef HEMV_LAUNCHER

template <typename Func, typename ScalarType, typename DataType>
inline void her(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, ScalarType alpha,
                sycl::buffer<DataType, 1> &x, int64_t incx, sycl::buffer<DataType, 1> &a,
                int64_t lda) {
    throw unimplemented("blas", "her", "for row_major layout");
}

#define HER_LAUNCHER(SCALAR_TYPE, DATA_TYPE, ROCBLAS_ROUTINE)                            \
    void her(sycl::queue &queue, uplo upper_lower, int64_t n, SCALAR_TYPE alpha,         \
             sycl::buffer<DATA_TYPE, 1> &x, int64_t incx, sycl::buffer<DATA_TYPE, 1> &a, \
             int64_t lda) {                                                              \
        her(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, lda);             \
    }

HER_LAUNCHER(float, std::complex<float>, rocblas_cher)
HER_LAUNCHER(double, std::complex<double>, rocblas_zher)

#undef HER_LAUNCHER

template <typename Func, typename T>
inline void her2(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 sycl::buffer<T, 1> &x, int64_t incx, sycl::buffer<T, 1> &y, int64_t incy,
                 sycl::buffer<T, 1> &a, int64_t lda) {
    throw unimplemented("blas", "her2", "for row_major layout");
}

#define HER2_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                  \
    void her2(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                    \
              sycl::buffer<TYPE, 1> &x, int64_t incx, sycl::buffer<TYPE, 1> &y, int64_t incy, \
              sycl::buffer<TYPE, 1> &a, int64_t lda) {                                        \
        her2(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);        \
    }

HER2_LAUNCHER(std::complex<float>, rocblas_cher2)
HER2_LAUNCHER(std::complex<double>, rocblas_zher2)

#undef HER2_LAUNCHER

template <typename Func, typename T>
inline void hpmv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 sycl::buffer<T, 1> &a, sycl::buffer<T, 1> &x, int64_t incx, T beta,
                 sycl::buffer<T, 1> &y, int64_t incy) {
    throw unimplemented("blas", "hpmv", "for row_major layout");
}

#define HPMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                               \
    void hpmv(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                 \
              sycl::buffer<TYPE, 1> &a, sycl::buffer<TYPE, 1> &x, int64_t incx, TYPE beta, \
              sycl::buffer<TYPE, 1> &y, int64_t incy) {                                    \
        hpmv(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);    \
    }

HPMV_LAUNCHER(std::complex<float>, rocblas_chpmv)
HPMV_LAUNCHER(std::complex<double>, rocblas_zhpmv)

#undef HPMV_LAUNCHER

template <typename Func, typename ScalarType, typename DataType>
inline void hpr(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, ScalarType alpha,
                sycl::buffer<DataType, 1> &x, int64_t incx, sycl::buffer<DataType, 1> &a) {
    throw unimplemented("blas", "hpr", "for row_major layout");
}

#define HPR_LAUNCHER(SCALAR_TYPE, DATA_TYPE, ROCBLAS_ROUTINE)                              \
    void hpr(sycl::queue &queue, uplo upper_lower, int64_t n, SCALAR_TYPE alpha,           \
             sycl::buffer<DATA_TYPE, 1> &x, int64_t incx, sycl::buffer<DATA_TYPE, 1> &a) { \
        hpr(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a);                    \
    }

HPR_LAUNCHER(float, std::complex<float>, rocblas_chpr)
HPR_LAUNCHER(double, std::complex<double>, rocblas_zhpr)

#undef HPR_LAUNCHER

template <typename Func, typename T>
inline void hpr2(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 sycl::buffer<T, 1> &x, int64_t incx, sycl::buffer<T, 1> &y, int64_t incy,
                 sycl::buffer<T, 1> &a) {
    throw unimplemented("blas", "hpr2", "for row_major layout");
}

#define HPR2_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                  \
    void hpr2(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                    \
              sycl::buffer<TYPE, 1> &x, int64_t incx, sycl::buffer<TYPE, 1> &y, int64_t incy, \
              sycl::buffer<TYPE, 1> &a) {                                                     \
        hpr2(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a);             \
    }

HPR2_LAUNCHER(std::complex<float>, rocblas_chpr2)
HPR2_LAUNCHER(std::complex<double>, rocblas_zhpr2)

#undef HPR2_LAUNCHER

template <typename Func, typename T>
inline void sbmv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, T alpha,
                 sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x, int64_t incx, T beta,
                 sycl::buffer<T, 1> &y, int64_t incy) {
    throw unimplemented("blas", "sbmv", "for row_major layout");
}

#define SBMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void sbmv(sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,           \
              sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x, int64_t incx,    \
              TYPE beta, sycl::buffer<TYPE, 1> &y, int64_t incy) {                              \
        sbmv(ROCBLAS_ROUTINE, queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy); \
    }

SBMV_LAUNCHER(float, rocblas_ssbmv)
SBMV_LAUNCHER(double, rocblas_dsbmv)

#undef SBMV_LAUNCHER

template <typename Func, typename T>
inline void symv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x, int64_t incx, T beta,
                 sycl::buffer<T, 1> &y, int64_t incy) {
    throw unimplemented("blas", "symv", "for row_major layout");
}

#define SYMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                 \
    void symv(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                   \
              sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x, int64_t incx, \
              TYPE beta, sycl::buffer<TYPE, 1> &y, int64_t incy) {                           \
        symv(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy); \
    }

SYMV_LAUNCHER(float, rocblas_ssymv)
SYMV_LAUNCHER(double, rocblas_dsymv)

#undef SYMV_LAUNCHER

template <typename Func, typename T>
inline void syr(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                sycl::buffer<T, 1> &x, int64_t incx, sycl::buffer<T, 1> &a, int64_t lda) {
    throw unimplemented("blas", "syr", "for row_major layout");
}

#define SYR_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                   \
    void syr(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                     \
             sycl::buffer<TYPE, 1> &x, int64_t incx, sycl::buffer<TYPE, 1> &a, int64_t lda) { \
        syr(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, lda);                  \
    }

SYR_LAUNCHER(float, rocblas_ssyr)
SYR_LAUNCHER(double, rocblas_dsyr)
// Intel does not support the following two
SYR_LAUNCHER(std::complex<float>, rocblas_csyr)
SYR_LAUNCHER(std::complex<double>, rocblas_zsyr)
#undef SYR_LAUNCHER

template <typename Func, typename T>
inline void syr2(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 sycl::buffer<T, 1> &x, int64_t incx, sycl::buffer<T, 1> &y, int64_t incy,
                 sycl::buffer<T, 1> &a, int64_t lda) {
    throw unimplemented("blas", "syr2", "for row_major layout");
}

#define SYR2_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                  \
    void syr2(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                    \
              sycl::buffer<TYPE, 1> &x, int64_t incx, sycl::buffer<TYPE, 1> &y, int64_t incy, \
              sycl::buffer<TYPE, 1> &a, int64_t lda) {                                        \
        syr2(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);        \
    }

SYR2_LAUNCHER(float, rocblas_ssyr2)
SYR2_LAUNCHER(double, rocblas_dsyr2)
// Intel does not support the following two
SYR2_LAUNCHER(std::complex<float>, rocblas_csyr2)
SYR2_LAUNCHER(std::complex<double>, rocblas_zsyr2)

#undef SYR2_LAUNCHER

template <typename Func, typename T>
inline void spmv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 sycl::buffer<T, 1> &a, sycl::buffer<T, 1> &x, int64_t incx, T beta,
                 sycl::buffer<T, 1> &y, int64_t incy) {
    throw unimplemented("blas", "spmv", "for row_major layout");
}

#define SPMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                               \
    void spmv(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                 \
              sycl::buffer<TYPE, 1> &a, sycl::buffer<TYPE, 1> &x, int64_t incx, TYPE beta, \
              sycl::buffer<TYPE, 1> &y, int64_t incy) {                                    \
        spmv(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);    \
    }

SPMV_LAUNCHER(float, rocblas_sspmv)
SPMV_LAUNCHER(double, rocblas_dspmv)

#undef SPMV_LAUNCHER

template <typename Func, typename T>
inline void spr(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                sycl::buffer<T, 1> &x, int64_t incx, sycl::buffer<T, 1> &a) {
    throw unimplemented("blas", "spr", "for row_major layout");
}

#define SPR_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                      \
    void spr(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,        \
             sycl::buffer<TYPE, 1> &x, int64_t incx, sycl::buffer<TYPE, 1> &a) { \
        spr(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a);          \
    }

SPR_LAUNCHER(float, rocblas_sspr)
SPR_LAUNCHER(double, rocblas_dspr)

#undef SPR_LAUNCHER

template <typename Func, typename T>
inline void spr2(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                 sycl::buffer<T, 1> &x, int64_t incx, sycl::buffer<T, 1> &y, int64_t incy,
                 sycl::buffer<T, 1> &a) {
    throw unimplemented("blas", "spr2", "for row_major layout");
}

#define SPR2_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                  \
    void spr2(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha,                    \
              sycl::buffer<TYPE, 1> &x, int64_t incx, sycl::buffer<TYPE, 1> &y, int64_t incy, \
              sycl::buffer<TYPE, 1> &a) {                                                     \
        spr2(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a);             \
    }

SPR2_LAUNCHER(float, rocblas_sspr2)
SPR2_LAUNCHER(double, rocblas_dspr2)

#undef SPR2_LAUNCHER

template <typename Func, typename T>
inline void tbmv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                 int64_t n, int64_t k, sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x,
                 int64_t incx) {
    throw unimplemented("blas", "tbmv", "for row_major layout");
}

#define TBMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void tbmv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              int64_t k, sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x,       \
              int64_t incx) {                                                                   \
        tbmv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);     \
    }

TBMV_LAUNCHER(float, rocblas_stbmv)
TBMV_LAUNCHER(double, rocblas_dtbmv)
TBMV_LAUNCHER(std::complex<float>, rocblas_ctbmv)
TBMV_LAUNCHER(std::complex<double>, rocblas_ztbmv)

#undef TBMV_LAUNCHER

template <typename Func, typename T>
inline void tbsv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                 int64_t n, int64_t k, sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x,
                 int64_t incx) {
    throw unimplemented("blas", "tbsv", "for row_major layout");
}

#define TBSV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void tbsv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              int64_t k, sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x,       \
              int64_t incx) {                                                                   \
        tbsv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);     \
    }

TBSV_LAUNCHER(float, rocblas_stbsv)
TBSV_LAUNCHER(double, rocblas_dtbsv)
TBSV_LAUNCHER(std::complex<float>, rocblas_ctbsv)
TBSV_LAUNCHER(std::complex<double>, rocblas_ztbsv)

#undef TBSV_LAUNCHER

template <typename Func, typename T>
inline void tpmv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                 int64_t n, sycl::buffer<T, 1> &a, sycl::buffer<T, 1> &x, int64_t incx) {
    throw unimplemented("blas", "tpmv", "for row_major layout");
}

#define TPMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void tpmv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              sycl::buffer<TYPE, 1> &a, sycl::buffer<TYPE, 1> &x, int64_t incx) {               \
        tpmv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, x, incx);             \
    }

TPMV_LAUNCHER(float, rocblas_stpmv)
TPMV_LAUNCHER(double, rocblas_dtpmv)
TPMV_LAUNCHER(std::complex<float>, rocblas_ctpmv)
TPMV_LAUNCHER(std::complex<double>, rocblas_ztpmv)

#undef TPMV_LAUNCHER

template <typename Func, typename T>
inline void tpsv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                 int64_t n, sycl::buffer<T, 1> &a, sycl::buffer<T, 1> &x, int64_t incx) {
    throw unimplemented("blas", "tpsv", "for row_major layout");
}

#define TPSV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void tpsv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              sycl::buffer<TYPE, 1> &a, sycl::buffer<TYPE, 1> &x, int64_t incx) {               \
        tpsv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, x, incx);             \
    }

TPSV_LAUNCHER(float, rocblas_stpsv)
TPSV_LAUNCHER(double, rocblas_dtpsv)
TPSV_LAUNCHER(std::complex<float>, rocblas_ctpsv)
TPSV_LAUNCHER(std::complex<double>, rocblas_ztpsv)

#undef TPSV_LAUNCHER

template <typename Func, typename T>
inline void trmv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                 int64_t n, sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x,
                 int64_t incx) {
    throw unimplemented("blas", "trmv", "for row_major layout");
}

#define TRMV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void trmv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x, int64_t incx) {  \
        trmv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);        \
    }

TRMV_LAUNCHER(float, rocblas_strmv)
TRMV_LAUNCHER(double, rocblas_dtrmv)
TRMV_LAUNCHER(std::complex<float>, rocblas_ctrmv)
TRMV_LAUNCHER(std::complex<double>, rocblas_ztrmv)

#undef TRMV_LAUNCHER

template <typename Func, typename T>
inline void trsv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                 int64_t n, sycl::buffer<T, 1> &a, int64_t lda, sycl::buffer<T, 1> &x,
                 int64_t incx) {
    throw unimplemented("blas", "trsv", "for row_major layout");
}

#define TRSV_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void trsv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              sycl::buffer<TYPE, 1> &a, int64_t lda, sycl::buffer<TYPE, 1> &x, int64_t incx) {  \
        trsv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);        \
    }

TRSV_LAUNCHER(float, rocblas_strsv)
TRSV_LAUNCHER(double, rocblas_dtrsv)
TRSV_LAUNCHER(std::complex<float>, rocblas_ctrsv)
TRSV_LAUNCHER(std::complex<double>, rocblas_ztrsv)

#undef TRSV_LAUNCHER

// USM APIs

template <typename Func, typename T>
inline sycl::event gemv(Func func, sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                        T alpha, const T *a, int64_t lda, const T *x, int64_t incx, T beta, T *y,
                        int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemv", "for row_major layout");
}

#define GEMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event gemv(sycl::queue &queue, transpose trans, int64_t m, int64_t n, TYPE alpha,       \
                     const TYPE *a, int64_t lda, const TYPE *x, int64_t incx, TYPE beta, TYPE *y, \
                     int64_t incy, const std::vector<sycl::event> &dependencies) {                \
        return gemv(ROCBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy,   \
                    dependencies);                                                                \
    }

GEMV_LAUNCHER_USM(float, rocblas_sgemv)
GEMV_LAUNCHER_USM(double, rocblas_dgemv)
GEMV_LAUNCHER_USM(std::complex<float>, rocblas_cgemv)
GEMV_LAUNCHER_USM(std::complex<double>, rocblas_zgemv)
#undef GEMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event gbmv(Func func, sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                        int64_t kl, int64_t ku, T alpha, const T *a, int64_t lda, const T *x,
                        int64_t incx, T beta, T *y, int64_t incy,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gbmv", "for row_major layout");
}

#define GBMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event gbmv(sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl,       \
                     int64_t ku, TYPE alpha, const TYPE *a, int64_t lda, const TYPE *x,           \
                     int64_t incx, TYPE beta, TYPE *y, int64_t incy,                              \
                     const std::vector<sycl::event> &dependencies) {                              \
        return gbmv(ROCBLAS_ROUTINE, queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, \
                    incy, dependencies);                                                          \
    }

GBMV_LAUNCHER_USM(float, rocblas_sgbmv)
GBMV_LAUNCHER_USM(double, rocblas_dgbmv)
GBMV_LAUNCHER_USM(std::complex<float>, rocblas_cgbmv)
GBMV_LAUNCHER_USM(std::complex<double>, rocblas_zgbmv)
#undef GBMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event ger(Func func, sycl::queue &queue, int64_t m, int64_t n, T alpha, const T *x,
                       int64_t incx, const T *y, int64_t incy, T *a, int64_t lda,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "ger", "for row_major layout");
}

#define GER_LAUNCHER_USM(EXT, TYPE, ROCBLAS_ROUTINE)                                             \
    sycl::event ger##EXT(sycl::queue &queue, int64_t m, int64_t n, TYPE alpha, const TYPE *x,    \
                         int64_t incx, const TYPE *y, int64_t incy, TYPE *a, int64_t lda,        \
                         const std::vector<sycl::event> &dependencies) {                         \
        return ger(ROCBLAS_ROUTINE, queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies); \
    }

GER_LAUNCHER_USM(, float, rocblas_sger)
GER_LAUNCHER_USM(, double, rocblas_dger)
GER_LAUNCHER_USM(u, std::complex<float>, rocblas_cgeru)
GER_LAUNCHER_USM(u, std::complex<double>, rocblas_zgeru)
GER_LAUNCHER_USM(c, std::complex<float>, rocblas_cgerc)
GER_LAUNCHER_USM(c, std::complex<double>, rocblas_zgerc)
#undef GER_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event hbmv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k,
                        T alpha, const T *a, int64_t lda, const T *x, int64_t incx, T beta, T *y,
                        int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "hbmv", "for row_major layout");
}

#define HBMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event hbmv(sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,      \
                     const TYPE *a, int64_t lda, const TYPE *x, int64_t incx, TYPE beta, TYPE *y, \
                     int64_t incy, const std::vector<sycl::event> &dependencies) {                \
        return hbmv(ROCBLAS_ROUTINE, queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,   \
                    incy, dependencies);                                                          \
    }

HBMV_LAUNCHER_USM(std::complex<float>, rocblas_chbmv)
HBMV_LAUNCHER_USM(std::complex<double>, rocblas_zhbmv)
#undef HBMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event hemv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                        const T *a, int64_t lda, const T *x, int64_t incx, T beta, T *y,
                        int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "hemv", "for row_major layout");
}

#define HEMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                   \
    sycl::event hemv(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *a,   \
                     int64_t lda, const TYPE *x, int64_t incx, TYPE beta, TYPE *y, int64_t incy,   \
                     const std::vector<sycl::event> &dependencies) {                               \
        return hemv(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, \
                    dependencies);                                                                 \
    }

HEMV_LAUNCHER_USM(std::complex<float>, rocblas_chemv)
HEMV_LAUNCHER_USM(std::complex<double>, rocblas_zhemv)
#undef HEMV_LAUNCHER_USM

template <typename Func, typename ScalarType, typename DataType>
inline sycl::event her(Func func, sycl::queue &queue, uplo upper_lower, int64_t n,
                       const ScalarType alpha, const DataType *x, int64_t incx, DataType *a,
                       int64_t lda, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "her", "for row_major layout");
}

#define HER_LAUNCHER_USM(SCALAR_TYPE, DATA_TYPE, ROCBLAS_ROUTINE)                                 \
    sycl::event her(sycl::queue &queue, uplo upper_lower, int64_t n, const SCALAR_TYPE alpha,     \
                    const DATA_TYPE *x, int64_t incx, DATA_TYPE *a, int64_t lda,                  \
                    const std::vector<sycl::event> &dependencies) {                               \
        return her(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, lda, dependencies); \
    }

HER_LAUNCHER_USM(float, std::complex<float>, rocblas_cher)
HER_LAUNCHER_USM(double, std::complex<double>, rocblas_zher)

#undef HER_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event her2(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                        const T *x, int64_t incx, const T *y, int64_t incy, T *a, int64_t lda,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "her2", "for row_major layout");
}

#define HER2_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                 \
    sycl::event her2(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *x, \
                     int64_t incx, const TYPE *y, int64_t incy, TYPE *a, int64_t lda,            \
                     const std::vector<sycl::event> &dependencies) {                             \
        return her2(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a, lda,     \
                    dependencies);                                                               \
    }

HER2_LAUNCHER_USM(std::complex<float>, rocblas_cher2)
HER2_LAUNCHER_USM(std::complex<double>, rocblas_zher2)

#undef HER2_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event hpmv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                        const T *a, const T *x, int64_t incx, T beta, T *y, int64_t incy,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "hpmv", "for row_major layout");
}

#define HPMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                 \
    sycl::event hpmv(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *a, \
                     const TYPE *x, int64_t incx, TYPE beta, TYPE *y, int64_t incy,              \
                     const std::vector<sycl::event> &dependencies) {                             \
        return hpmv(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, a, x, incx, beta, y, incy,    \
                    dependencies);                                                               \
    }

HPMV_LAUNCHER_USM(std::complex<float>, rocblas_chpmv)
HPMV_LAUNCHER_USM(std::complex<double>, rocblas_zhpmv)

#undef HPMV_LAUNCHER_USM

template <typename Func, typename ScalarType, typename DataType>
inline sycl::event hpr(Func func, sycl::queue &queue, uplo upper_lower, int64_t n,
                       const ScalarType alpha, const DataType *x, int64_t incx, DataType *a,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "hpr", "for row_major layout");
}

#define HPR_LAUNCHER_USM(SCALAR_TYPE, DATA_TYPE, ROCBLAS_ROUTINE)                             \
    sycl::event hpr(sycl::queue &queue, uplo upper_lower, int64_t n, const SCALAR_TYPE alpha, \
                    const DATA_TYPE *x, int64_t incx, DATA_TYPE *a,                           \
                    const std::vector<sycl::event> &dependencies) {                           \
        return hpr(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, dependencies);  \
    }

HPR_LAUNCHER_USM(float, std::complex<float>, rocblas_chpr)
HPR_LAUNCHER_USM(double, std::complex<double>, rocblas_zhpr)

#undef HPR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event hpr2(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                        const T *x, int64_t incx, const T *y, int64_t incy, T *a,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "hpr2", "for row_major layout");
}

#define HPR2_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                 \
    sycl::event hpr2(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *x, \
                     int64_t incx, const TYPE *y, int64_t incy, TYPE *a,                         \
                     const std::vector<sycl::event> &dependencies) {                             \
        return hpr2(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a,          \
                    dependencies);                                                               \
    }

HPR2_LAUNCHER_USM(std::complex<float>, rocblas_chpr2)
HPR2_LAUNCHER_USM(std::complex<double>, rocblas_zhpr2)

#undef HPR2_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event sbmv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k,
                        T alpha, const T *a, int64_t lda, const T *x, int64_t incx, T beta, T *y,
                        int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "sbmv", "for row_major layout");
}

#define SBMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event sbmv(sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,      \
                     const TYPE *a, int64_t lda, const TYPE *x, int64_t incx, TYPE beta, TYPE *y, \
                     int64_t incy, const std::vector<sycl::event> &dependencies) {                \
        return sbmv(ROCBLAS_ROUTINE, queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,   \
                    incy, dependencies);                                                          \
    }

SBMV_LAUNCHER_USM(float, rocblas_ssbmv)
SBMV_LAUNCHER_USM(double, rocblas_dsbmv)

#undef SBMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event symv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                        const T *a, int64_t lda, const T *x, int64_t incx, T beta, T *y,
                        int64_t incy, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "symv", "for row_major layout");
}

#define SYMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                   \
    sycl::event symv(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *a,   \
                     int64_t lda, const TYPE *x, int64_t incx, TYPE beta, TYPE *y, int64_t incy,   \
                     const std::vector<sycl::event> &dependencies) {                               \
        return symv(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, \
                    dependencies);                                                                 \
    }

SYMV_LAUNCHER_USM(float, rocblas_ssymv)
SYMV_LAUNCHER_USM(double, rocblas_dsymv)

#undef SYMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event syr(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                       const T *x, int64_t incx, T *a, int64_t lda,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syr", "for row_major layout");
}

#define SYR_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                   \
    sycl::event syr(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *x,   \
                    int64_t incx, TYPE *a, int64_t lda,                                           \
                    const std::vector<sycl::event> &dependencies) {                               \
        return syr(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, lda, dependencies); \
    }

SYR_LAUNCHER_USM(float, rocblas_ssyr)
SYR_LAUNCHER_USM(double, rocblas_dsyr)
// Intel does not support the following two
SYR_LAUNCHER_USM(std::complex<float>, rocblas_csyr)
SYR_LAUNCHER_USM(std::complex<double>, rocblas_zsyr)
#undef SYR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event syr2(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                        const T *x, int64_t incx, const T *y, int64_t incy, T *a, int64_t lda,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syr2", "for row_major layout");
}

#define SYR2_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                 \
    sycl::event syr2(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *x, \
                     int64_t incx, const TYPE *y, int64_t incy, TYPE *a, int64_t lda,            \
                     const std::vector<sycl::event> &dependencies) {                             \
        return syr2(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a, lda,     \
                    dependencies);                                                               \
    }

SYR2_LAUNCHER_USM(float, rocblas_ssyr2)
SYR2_LAUNCHER_USM(double, rocblas_dsyr2)
// Intel does not support the following two
SYR2_LAUNCHER_USM(std::complex<float>, rocblas_csyr2)
SYR2_LAUNCHER_USM(std::complex<double>, rocblas_zsyr2)

#undef SYR2_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event spmv(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                        const T *a, const T *x, int64_t incx, T beta, T *y, int64_t incy,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "spmv", "for row_major layout");
}

#define SPMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                 \
    sycl::event spmv(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *a, \
                     const TYPE *x, int64_t incx, TYPE beta, TYPE *y, int64_t incy,              \
                     const std::vector<sycl::event> &dependencies) {                             \
        return spmv(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, a, x, incx, beta, y, incy,    \
                    dependencies);                                                               \
    }

SPMV_LAUNCHER_USM(float, rocblas_sspmv)
SPMV_LAUNCHER_USM(double, rocblas_dspmv)

#undef SPMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event spr(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                       const T *x, int64_t incx, T *a,
                       const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "spr", "for row_major layout");
}

#define SPR_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                 \
    sycl::event spr(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *x, \
                    int64_t incx, TYPE *a, const std::vector<sycl::event> &dependencies) {      \
        return spr(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, dependencies);    \
    }

SPR_LAUNCHER_USM(float, rocblas_sspr)
SPR_LAUNCHER_USM(double, rocblas_dspr)

#undef SPR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event spr2(Func func, sycl::queue &queue, uplo upper_lower, int64_t n, T alpha,
                        const T *x, int64_t incx, const T *y, int64_t incy, T *a,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "spr2", "for row_major layout");
}

#define SPR2_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                 \
    sycl::event spr2(sycl::queue &queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE *x, \
                     int64_t incx, const TYPE *y, int64_t incy, TYPE *a,                         \
                     const std::vector<sycl::event> &dependencies) {                             \
        return spr2(ROCBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a,          \
                    dependencies);                                                               \
    }

SPR2_LAUNCHER_USM(float, rocblas_sspr2)
SPR2_LAUNCHER_USM(double, rocblas_dspr2)

#undef SPR2_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event tbmv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans,
                        diag unit_diag, int64_t n, int64_t k, const T *a, int64_t lda, T *x,
                        int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tbmv", "for row_major layout");
}

#define TBMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event tbmv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,       \
                     int64_t n, int64_t k, const TYPE *a, int64_t lda, TYPE *x, int64_t incx,     \
                     const std::vector<sycl::event> &dependencies) {                              \
        return tbmv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, \
                    dependencies);                                                                \
    }

TBMV_LAUNCHER_USM(float, rocblas_stbmv)
TBMV_LAUNCHER_USM(double, rocblas_dtbmv)
TBMV_LAUNCHER_USM(std::complex<float>, rocblas_ctbmv)
TBMV_LAUNCHER_USM(std::complex<double>, rocblas_ztbmv)

#undef TBMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event tbsv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans,
                        diag unit_diag, int64_t n, int64_t k, const T *a, int64_t lda, T *x,
                        int64_t incx, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tbsv", "for row_major layout");
}

#define TBSV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event tbsv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,       \
                     int64_t n, int64_t k, const TYPE *a, int64_t lda, TYPE *x, int64_t incx,     \
                     const std::vector<sycl::event> &dependencies) {                              \
        return tbsv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, \
                    dependencies);                                                                \
    }

TBSV_LAUNCHER_USM(float, rocblas_stbsv)
TBSV_LAUNCHER_USM(double, rocblas_dtbsv)
TBSV_LAUNCHER_USM(std::complex<float>, rocblas_ctbsv)
TBSV_LAUNCHER_USM(std::complex<double>, rocblas_ztbsv)

#undef TBSV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event tpmv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans,
                        diag unit_diag, int64_t n, const T *a, T *x, int64_t incx,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tpmv", "for row_major layout");
}

#define TPMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                            \
    sycl::event tpmv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, \
                     int64_t n, const TYPE *a, TYPE *x, int64_t incx,                       \
                     const std::vector<sycl::event> &dependencies) {                        \
        return tpmv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, x, incx,   \
                    dependencies);                                                          \
    }

TPMV_LAUNCHER_USM(float, rocblas_stpmv)
TPMV_LAUNCHER_USM(double, rocblas_dtpmv)
TPMV_LAUNCHER_USM(std::complex<float>, rocblas_ctpmv)
TPMV_LAUNCHER_USM(std::complex<double>, rocblas_ztpmv)

#undef TPMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event tpsv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans,
                        diag unit_diag, int64_t n, const T *a, T *x, int64_t incx,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "tpsv", "for row_major layout");
}

#define TPSV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                            \
    sycl::event tpsv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, \
                     int64_t n, const TYPE *a, TYPE *x, int64_t incx,                       \
                     const std::vector<sycl::event> &dependencies) {                        \
        return tpsv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, x, incx,   \
                    dependencies);                                                          \
    }

TPSV_LAUNCHER_USM(float, rocblas_stpsv)
TPSV_LAUNCHER_USM(double, rocblas_dtpsv)
TPSV_LAUNCHER_USM(std::complex<float>, rocblas_ctpsv)
TPSV_LAUNCHER_USM(std::complex<double>, rocblas_ztpsv)

#undef TPSV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event trmv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans,
                        diag unit_diag, int64_t n, const T *a, int64_t lda, T *x, int64_t incx,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trmv", "for row_major layout");
}

#define TRMV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                               \
    sycl::event trmv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,    \
                     int64_t n, const TYPE *a, int64_t lda, TYPE *x, int64_t incx,             \
                     const std::vector<sycl::event> &dependencies) {                           \
        return trmv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, \
                    dependencies);                                                             \
    }

TRMV_LAUNCHER_USM(float, rocblas_strmv)
TRMV_LAUNCHER_USM(double, rocblas_dtrmv)
TRMV_LAUNCHER_USM(std::complex<float>, rocblas_ctrmv)
TRMV_LAUNCHER_USM(std::complex<double>, rocblas_ztrmv)

#undef TRMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event trsv(Func func, sycl::queue &queue, uplo upper_lower, transpose trans,
                        diag unit_diag, int64_t n, const T *a, int64_t lda, T *x, int64_t incx,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsv", "for row_major layout");
}

#define TRSV_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                               \
    sycl::event trsv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,    \
                     int64_t n, const TYPE *a, int64_t lda, TYPE *x, int64_t incx,             \
                     const std::vector<sycl::event> &dependencies) {                           \
        return trsv(ROCBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, \
                    dependencies);                                                             \
    }

TRSV_LAUNCHER_USM(float, rocblas_strsv)
TRSV_LAUNCHER_USM(double, rocblas_dtrsv)
TRSV_LAUNCHER_USM(std::complex<float>, rocblas_ctrsv)
TRSV_LAUNCHER_USM(std::complex<double>, rocblas_ztrsv)

#undef TRSV_LAUNCHER_USM

} // namespace row_major
} // namespace rocblas
} // namespace blas
} // namespace mkl
} // namespace oneapi
