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
inline void copy_batch(Func func, sycl::queue &queue, int64_t n, sycl::buffer<T, 1> &x,
                       int64_t incx, int64_t stridex, sycl::buffer<T, 1> &y, int64_t incy,
                       int64_t stridey, int64_t batch_size) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy, stridex, stridey, batch_size);

    queue.submit([&](sycl::handler &cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            auto y_ = sc.get_mem<rocDataType *>(y_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, n, x_, incx, stridex, y_, incy, stridey,
                                    batch_size);
        });
    });
}

#define COPY_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                     \
    void copy_batch(sycl::queue &queue, int64_t n, sycl::buffer<TYPE, 1> &x, int64_t incx,     \
                    int64_t stridex, sycl::buffer<TYPE, 1> &y, int64_t incy, int64_t stridey,  \
                    int64_t batch_size) {                                                      \
        copy_batch(ROCBLAS_ROUTINE, queue, n, x, incx, stridex, y, incy, stridey, batch_size); \
    }

COPY_STRIDED_BATCH_LAUNCHER(float, rocblas_scopy_strided_batched)
COPY_STRIDED_BATCH_LAUNCHER(double, rocblas_dcopy_strided_batched)
COPY_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_ccopy_strided_batched)
COPY_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_zcopy_strided_batched)

#undef COPY_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void axpy_batch(Func func, sycl::queue &queue, int64_t n, T alpha, sycl::buffer<T, 1> &x,
                       int64_t incx, int64_t stridex, sycl::buffer<T, 1> &y, int64_t incy,
                       int64_t stridey, int64_t batch_size) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy, stridex, stridey, batch_size);

    queue.submit([&](sycl::handler &cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = sc.get_mem<rocDataType *>(x_acc);
            auto y_ = sc.get_mem<rocDataType *>(y_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, n, (rocDataType *)&alpha, x_, incx, stridex,
                                    y_, incy, stridey, batch_size);
        });
    });
}

#define AXPY_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                 \
    void axpy_batch(sycl::queue &queue, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1> &x,   \
                    int64_t incx, int64_t stridex, sycl::buffer<TYPE, 1> &y, int64_t incy, \
                    int64_t stridey, int64_t batch_size) {                                 \
        axpy_batch(ROCBLAS_ROUTINE, queue, n, alpha, x, incx, stridex, y, incy, stridey,   \
                   batch_size);                                                            \
    }

AXPY_STRIDED_BATCH_LAUNCHER(float, rocblas_saxpy_strided_batched)
AXPY_STRIDED_BATCH_LAUNCHER(double, rocblas_daxpy_strided_batched)
AXPY_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_caxpy_strided_batched)
AXPY_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_zaxpy_strided_batched)

#undef AXPY_BATCH_LAUNCHER

template <typename Func, typename T>
inline void gemv_batch(Func func, sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                       T alpha, sycl::buffer<T, 1> &a, int64_t lda, int64_t stridea,
                       sycl::buffer<T, 1> &x, int64_t incx, int64_t stridex, T beta,
                       sycl::buffer<T, 1> &y, int64_t incy, int64_t stridey, int64_t batch_size) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, lda, incx, incy, stridea, stridex, stridey, batch_size);

    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<const rocDataType *>(a_acc);
            auto x_ = sc.get_mem<const rocDataType *>(x_acc);
            auto y_ = sc.get_mem<rocDataType *>(y_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_operation(trans), m, n,
                                    (rocDataType *)&alpha, a_, lda, stridea, x_, incx, stridex,
                                    (rocDataType *)&beta, y_, incy, stridey, batch_size);
        });
    });
}

#define GEMV_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                         \
    void gemv_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, TYPE alpha,         \
                    sycl::buffer<TYPE, 1> &a, int64_t lda, int64_t stridea,                        \
                    sycl::buffer<TYPE, 1> &x, int64_t incx, int64_t stridex, TYPE beta,            \
                    sycl::buffer<TYPE, 1> &y, int64_t incy, int64_t stridey, int64_t batch_size) { \
        gemv_batch(ROCBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, stridea, x, incx, stridex,  \
                   beta, y, incy, stridey, batch_size);                                            \
    }

GEMV_STRIDED_BATCH_LAUNCHER(float, rocblas_sgemv_strided_batched)
GEMV_STRIDED_BATCH_LAUNCHER(double, rocblas_dgemv_strided_batched)
GEMV_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_cgemv_strided_batched)
GEMV_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_zgemv_strided_batched)

#undef GEMV_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void dgmm_batch(Func func, sycl::queue &queue, side left_right, int64_t m, int64_t n,
                       sycl::buffer<T, 1> &a, int64_t lda, int64_t stridea, sycl::buffer<T, 1> &x,
                       int64_t incx, int64_t stridex, sycl::buffer<T, 1> &c, int64_t ldc,
                       int64_t stridec, int64_t batch_size) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldc, incx, stridea, stridex, stridec, batch_size);

    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<const rocDataType *>(a_acc);
            auto x_ = sc.get_mem<const rocDataType *>(x_acc);
            auto c_ = sc.get_mem<rocDataType *>(c_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_side_mode(left_right), m, n, a_,
                                    lda, stridea, x_, incx, stridex, c_, ldc, stridec, batch_size);
        });
    });
}

#define DGMM_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                         \
    void dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,                     \
                    sycl::buffer<TYPE, 1> &a, int64_t lda, int64_t stridea,                        \
                    sycl::buffer<TYPE, 1> &x, int64_t incx, int64_t stridex,                       \
                    sycl::buffer<TYPE, 1> &c, int64_t ldc, int64_t stridec, int64_t batch_size) {  \
        dgmm_batch(ROCBLAS_ROUTINE, queue, left_right, m, n, a, lda, stridea, x, incx, stridex, c, \
                   ldc, stridec, batch_size);                                                      \
    }

DGMM_STRIDED_BATCH_LAUNCHER(float, rocblas_sdgmm_strided_batched)
DGMM_STRIDED_BATCH_LAUNCHER(double, rocblas_ddgmm_strided_batched)
DGMM_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_cdgmm_strided_batched)
DGMM_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_zdgmm_strided_batched)

#undef DGMM_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void gemm_batch(Func func, sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                       int64_t n, int64_t k, T alpha, sycl::buffer<T, 1> &a, int64_t lda,
                       int64_t stridea, sycl::buffer<T, 1> &b, int64_t ldb, int64_t strideb, T beta,
                       sycl::buffer<T, 1> &c, int64_t ldc, int64_t stridec, int64_t batch_size) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, ldb, ldc, stridea, strideb, stridec, batch_size);

    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<const rocDataType *>(a_acc);
            auto b_ = sc.get_mem<const rocDataType *>(b_acc);
            auto c_ = sc.get_mem<rocDataType *>(c_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_operation(transa),
                                    get_rocblas_operation(transb), m, n, k, (rocDataType *)&alpha,
                                    a_, lda, stridea, b_, ldb, strideb, (rocDataType *)&beta, c_,
                                    ldc, stridec, batch_size);
        });
    });
}

#define GEMM_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                         \
    void gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,  \
                    int64_t k, TYPE alpha, sycl::buffer<TYPE, 1> &a, int64_t lda, int64_t stridea, \
                    sycl::buffer<TYPE, 1> &b, int64_t ldb, int64_t strideb, TYPE beta,             \
                    sycl::buffer<TYPE, 1> &c, int64_t ldc, int64_t stridec, int64_t batch_size) {  \
        gemm_batch(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, stridea, b,     \
                   ldb, strideb, beta, c, ldc, stridec, batch_size);                               \
    }

GEMM_STRIDED_BATCH_LAUNCHER(sycl::half, rocblas_hgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER(float, rocblas_sgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER(double, rocblas_dgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_cgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_zgemm_strided_batched)

#undef GEMM_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void trsm_batch(Func func, sycl::queue &queue, side left_right, uplo upper_lower,
                       transpose trans, diag unit_diag, int64_t m, int64_t n, T alpha,
                       sycl::buffer<T, 1> &a, int64_t lda, int64_t stridea, sycl::buffer<T, 1> &b,
                       int64_t ldb, int64_t strideb, int64_t batch_size) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, stridea, strideb, batch_size);

    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<const rocDataType *>(a_acc);
            auto b_ = sc.get_mem<rocDataType *>(b_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_side_mode(left_right),
                                    get_rocblas_fill_mode(upper_lower),
                                    get_rocblas_operation(trans), get_rocblas_diag_type(unit_diag),
                                    m, n, (rocDataType *)&alpha, a_, lda, stridea, b_, ldb, strideb,
                                    batch_size);
        });
    });
}

#define TRSM_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                         \
    void trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,        \
                    diag unit_diag, int64_t m, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1> &a,    \
                    int64_t lda, int64_t stridea, sycl::buffer<TYPE, 1> &b, int64_t ldb,           \
                    int64_t strideb, int64_t batch_size) {                                         \
        trsm_batch(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, \
                   a, lda, stridea, b, ldb, strideb, batch_size);                                  \
    }

TRSM_STRIDED_BATCH_LAUNCHER(float, rocblas_strsm_strided_batched)
TRSM_STRIDED_BATCH_LAUNCHER(double, rocblas_dtrsm_strided_batched)
TRSM_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_ctrsm_strided_batched)
TRSM_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_ztrsm_strided_batched)

#undef TRSM_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void syrk_batch(Func func, sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                       int64_t k, T alpha, sycl::buffer<T, 1> &a, int64_t lda, int64_t stridea,
                       T beta, sycl::buffer<T, 1> &c, int64_t ldc, int64_t stridec,
                       int64_t batch_size) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, k, lda, ldc, stridea, stridec, batch_size);

    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<const rocDataType *>(a_acc);
            auto c_ = sc.get_mem<rocDataType *>(c_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                    get_rocblas_operation(trans), n, k, (rocDataType *)&alpha, a_,
                                    lda, stridea, (rocDataType *)&beta, c_, ldc, stridec,
                                    batch_size);
        });
    });
}

#define SYRK_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                         \
    void syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,   \
                    TYPE alpha, sycl::buffer<TYPE, 1> &a, int64_t lda, int64_t stridea, TYPE beta, \
                    sycl::buffer<TYPE, 1> &c, int64_t ldc, int64_t stridec, int64_t batch_size) {  \
        syrk_batch(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, stridea, beta, \
                   c, ldc, stridec, batch_size);                                                   \
    }

SYRK_STRIDED_BATCH_LAUNCHER(float, rocblas_ssyrk_strided_batched)
SYRK_STRIDED_BATCH_LAUNCHER(double, rocblas_dsyrk_strided_batched)
SYRK_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_csyrk_strided_batched)
SYRK_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_zsyrk_strided_batched)

#undef SYRK_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void omatcopy_batch(Func func, sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                           const T alpha, sycl::buffer<T, 1> &a, int64_t lda, int64_t stridea,
                           sycl::buffer<T, 1> &b, int64_t ldb, int64_t strideb,
                           int64_t batch_size) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, stridea, strideb, batch_size);

    const T beta = 0;
    const int64_t new_m = trans == oneapi::mkl::transpose::nontrans ? m : n;
    const int64_t new_n = trans == oneapi::mkl::transpose::nontrans ? n : m;

    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<const rocDataType *>(a_acc);
            auto b_ = sc.get_mem<rocDataType *>(b_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_operation(trans),
                                    get_rocblas_operation(trans), new_m, new_n,
                                    (rocDataType *)&alpha, a_, lda, stridea, (rocDataType *)&beta,
                                    nullptr, lda, stridea, b_, ldb, strideb, batch_size);
        });
    });
}

#define OMATCOPY_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                    \
    void omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,                \
                        const TYPE alpha, sycl::buffer<TYPE, 1> &a, int64_t lda, int64_t stridea, \
                        sycl::buffer<TYPE, 1> &b, int64_t ldb, int64_t strideb,                   \
                        int64_t batch_size) {                                                     \
        omatcopy_batch(ROCBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, stridea, b, ldb,       \
                       strideb, batch_size);                                                      \
    }

OMATCOPY_STRIDED_BATCH_LAUNCHER(float, rocblas_sgeam_strided_batched)
OMATCOPY_STRIDED_BATCH_LAUNCHER(double, rocblas_dgeam_strided_batched)
OMATCOPY_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_cgeam_strided_batched)
OMATCOPY_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_zgeam_strided_batched)

#undef OMATCOPY_STRIDED_BATCH_LAUNCHER

void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                    sycl::buffer<float, 1> &ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size) {
    throw unimplemented("blas", "column_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                    sycl::buffer<double, 1> &ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size) {
    throw unimplemented("blas", "column_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
    throw unimplemented("blas", "column_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
    throw unimplemented("blas", "column_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

template <typename Func, typename T>
inline void omatadd_batch(Func func, sycl::queue &queue, transpose transa, transpose transb,
                          int64_t m, int64_t n, const T alpha, sycl::buffer<T, 1> &a, int64_t lda,
                          int64_t stridea, const T beta, sycl::buffer<T, 1> &b, int64_t ldb,
                          int64_t strideb, sycl::buffer<T, 1> &c, int64_t ldc, int64_t stridec,
                          int64_t batch_size) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, ldc, stridea, strideb, stridec, batch_size);

    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<const rocDataType *>(a_acc);
            auto b_ = sc.get_mem<const rocDataType *>(b_acc);
            auto c_ = sc.get_mem<rocDataType *>(c_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_operation(transa),
                                    get_rocblas_operation(transb), m, n, (rocDataType *)&alpha, a_,
                                    lda, stridea, (rocDataType *)&beta, b_, ldb, strideb, c_, ldc,
                                    stridec, batch_size);
        });
    });
}

#define OMATADD_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                     \
    void omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,         \
                       int64_t n, const TYPE alpha, sycl::buffer<TYPE, 1> &a, int64_t lda,        \
                       int64_t stridea, const TYPE beta, sycl::buffer<TYPE, 1> &b, int64_t ldb,   \
                       int64_t strideb, sycl::buffer<TYPE, 1> &c, int64_t ldc, int64_t stridec,   \
                       int64_t batch_size) {                                                      \
        omatadd_batch(ROCBLAS_ROUTINE, queue, transa, transb, m, n, alpha, a, lda, stridea, beta, \
                      b, ldb, strideb, c, ldc, stridec, batch_size);                              \
    }

OMATADD_STRIDED_BATCH_LAUNCHER(float, rocblas_sgeam_strided_batched)
OMATADD_STRIDED_BATCH_LAUNCHER(double, rocblas_dgeam_strided_batched)
OMATADD_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_cgeam_strided_batched)
OMATADD_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_zgeam_strided_batched)

#undef OMATADD_STRIDED_BATCH_LAUNCHER

// USM APIs

template <typename Func, typename T>
inline sycl::event copy_batch(Func func, sycl::queue &queue, int64_t *n, const T **x, int64_t *incx,
                              T **y, int64_t *incy, int64_t group_count, int64_t *group_size,
                              const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(n[i], incx[i], incy[i], group_size[i]);
    }

    auto done = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            int64_t offset = 0;
            rocblas_status err;
            for (int64_t i = 0; i < group_count; i++) {
                auto **x_ = reinterpret_cast<const rocDataType **>(x);
                auto **y_ = reinterpret_cast<rocDataType **>(y);
                ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, (int)n[i], x_ + offset, (int)incx[i],
                                        y_ + offset, (int)incy[i], (int)group_size[i]);
                offset += group_size[i];
            }
        });
    });

    return done;
}

#define COPY_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                          \
    sycl::event copy_batch(sycl::queue &queue, int64_t *n, const TYPE **x, int64_t *incx,       \
                           TYPE **y, int64_t *incy, int64_t group_count, int64_t *group_size,   \
                           const std::vector<sycl::event> &dependencies) {                      \
        return copy_batch(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy, group_count, group_size, \
                          dependencies);                                                        \
    }

COPY_BATCH_LAUNCHER_USM(float, rocblas_scopy_batched)
COPY_BATCH_LAUNCHER_USM(double, rocblas_dcopy_batched)
COPY_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_ccopy_batched)
COPY_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zcopy_batched)

#undef COPY_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event copy_batch(Func func, sycl::queue &queue, int64_t n, const T *x, int64_t incx,
                              int64_t stridex, T *y, int64_t incy, int64_t stridey,
                              int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy, stridex, stridey, batch_size);

    auto done = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<const rocDataType *>(x);
            auto y_ = reinterpret_cast<rocDataType *>(y);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, n, x_, incx, stridex, y_, incy, stridey,
                                    batch_size);
        });
    });

    return done;
}

#define COPY_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                 \
    sycl::event copy_batch(sycl::queue &queue, int64_t n, const TYPE *x, int64_t incx,         \
                           int64_t stridex, TYPE *y, int64_t incy, int64_t stridey,            \
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) { \
        return copy_batch(ROCBLAS_ROUTINE, queue, n, x, incx, stridex, y, incy, stridey,       \
                          batch_size, dependencies);                                           \
    }

COPY_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_scopy_strided_batched)
COPY_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_dcopy_strided_batched)
COPY_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_ccopy_strided_batched)
COPY_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zcopy_strided_batched)

#undef COPY_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event axpy_batch(Func func, sycl::queue &queue, int64_t *n, T *alpha, const T **x,
                              int64_t *incx, T **y, int64_t *incy, int64_t group_count,
                              int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(n[i], incx[i], incy[i], group_size[i]);
    }

    auto done = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            int64_t offset = 0;
            rocblas_status err;
            for (int64_t i = 0; i < group_count; i++) {
                auto **x_ = reinterpret_cast<const rocDataType **>(x);
                auto **y_ = reinterpret_cast<rocDataType **>(y);
                ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, (int)n[i], (rocDataType *)&alpha[i],
                                        x_ + offset, (int)incx[i], y_ + offset, (int)incy[i],
                                        (int)group_size[i]);
                offset += group_size[i];
            }
        });
    });

    return done;
}

#define AXPY_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                          \
    sycl::event axpy_batch(sycl::queue &queue, int64_t *n, TYPE *alpha, const TYPE **x,         \
                           int64_t *incx, TYPE **y, int64_t *incy, int64_t group_count,         \
                           int64_t *group_size, const std::vector<sycl::event> &dependencies) { \
        return axpy_batch(ROCBLAS_ROUTINE, queue, n, alpha, x, incx, y, incy, group_count,      \
                          group_size, dependencies);                                            \
    }

AXPY_BATCH_LAUNCHER_USM(float, rocblas_saxpy_batched)
AXPY_BATCH_LAUNCHER_USM(double, rocblas_daxpy_batched)
AXPY_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_caxpy_batched)
AXPY_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zaxpy_batched)

#undef AXPY_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event axpy_batch(Func func, sycl::queue &queue, int64_t n, T alpha, const T *x,
                              int64_t incx, int64_t stridex, T *y, int64_t incy, int64_t stridey,
                              int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy, stridex, stridey, batch_size);

    auto done = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<const rocDataType *>(x);
            auto y_ = reinterpret_cast<rocDataType *>(y);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, n, (rocDataType *)&alpha, x_, incx, stridex,
                                    y_, incy, stridey, batch_size);
        });
    });

    return done;
}

#define AXPY_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                     \
    sycl::event axpy_batch(sycl::queue &queue, int64_t n, TYPE alpha, const TYPE *x, int64_t incx, \
                           int64_t stridex, TYPE *y, int64_t incy, int64_t stridey,                \
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {     \
        return axpy_batch(ROCBLAS_ROUTINE, queue, n, alpha, x, incx, stridex, y, incy, stridey,    \
                          batch_size, dependencies);                                               \
    }

AXPY_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_saxpy_strided_batched)
AXPY_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_daxpy_strided_batched)
AXPY_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_caxpy_strided_batched)
AXPY_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zaxpy_strided_batched)

#undef AXPY_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event gemv_batch(Func func, sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                              T alpha, const T *a, int64_t lda, int64_t stridea, const T *x,
                              int64_t incx, int64_t stridex, T beta, T *y, int64_t incy,
                              int64_t stridey, int64_t batch_size,
                              const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, m, lda, incx, incy, stridea, stridex, stridey, batch_size);

    auto done = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto x_ = reinterpret_cast<const rocDataType *>(x);
            auto y_ = reinterpret_cast<rocDataType *>(y);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_operation(trans), m, n,
                                    (rocDataType *)&alpha, a_, lda, stridea, x_, incx, stridex,
                                    (rocDataType *)&beta, y_, incy, stridey, batch_size);
        });
    });

    return done;
}

#define GEMV_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                    \
    sycl::event gemv_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, TYPE alpha, \
                           const TYPE *a, int64_t lda, int64_t stridea, const TYPE *x,            \
                           int64_t incx, int64_t stridex, TYPE beta, TYPE *y, int64_t incy,       \
                           int64_t stridey, int64_t batch_size,                                   \
                           const std::vector<sycl::event> &dependencies) {                        \
        return gemv_batch(ROCBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, stridea, x, incx,   \
                          stridex, beta, y, incy, stridey, batch_size, dependencies);             \
    }

GEMV_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_sgemv_strided_batched)
GEMV_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_dgemv_strided_batched)
GEMV_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cgemv_strided_batched)
GEMV_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zgemv_strided_batched)

#undef GEMV_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event gemv_batch(Func func, sycl::queue &queue, transpose *trans, int64_t *m,
                              int64_t *n, T *alpha, const T **a, int64_t *lda, const T **x,
                              int64_t *incx, T *beta, T **y, int64_t *incy, int64_t group_count,
                              int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(m[i], n[i], lda[i], incx[i], incy[i], group_size[i]);
    }

    auto done = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            int64_t offset = 0;
            rocblas_status err;
            for (int64_t i = 0; i < group_count; i++) {
                auto **a_ = reinterpret_cast<const rocDataType **>(a);
                auto **x_ = reinterpret_cast<const rocDataType **>(x);
                auto **y_ = reinterpret_cast<rocDataType **>(y);
                ROCBLAS_ERROR_FUNC_SYNC(
                    func, err, handle, get_rocblas_operation(trans[i]), (int)m[i], (int)n[i],
                    (rocDataType *)&alpha[i], a_ + offset, (int)lda[i], x_ + offset, (int)incx[i],
                    (rocDataType *)&beta[i], y_ + offset, (int)incy[i], (int)group_size[i]);
                offset += group_size[i];
            }
        });
    });

    return done;
}

#define GEMV_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                             \
    sycl::event gemv_batch(                                                                        \
        sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n, TYPE *alpha, const TYPE **a, \
        int64_t *lda, const TYPE **x, int64_t *incx, TYPE *beta, TYPE **y, int64_t *incy,          \
        int64_t group_count, int64_t *group_size, const std::vector<sycl::event> &dependencies) {  \
        return gemv_batch(ROCBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, x, incx, beta, y,    \
                          incy, group_count, group_size, dependencies);                            \
    }

GEMV_BATCH_LAUNCHER_USM(float, rocblas_sgemv_batched)
GEMV_BATCH_LAUNCHER_USM(double, rocblas_dgemv_batched)
GEMV_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cgemv_batched)
GEMV_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zgemv_batched)

#undef GEMV_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event dgmm_batch(Func func, sycl::queue &queue, side left_right, int64_t m, int64_t n,
                              const T *a, int64_t lda, int64_t stridea, const T *x, int64_t incx,
                              int64_t stridex, T *c, int64_t ldc, int64_t stridec,
                              int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, incx, stridea, stridex, stridec, batch_size);

    auto done = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto x_ = reinterpret_cast<const rocDataType *>(x);
            auto c_ = reinterpret_cast<rocDataType *>(c);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_side_mode(left_right), m, n, a_,
                                    lda, stridea, x_, incx, stridex, c_, ldc, stridec, batch_size);
        });
    });

    return done;
}

#define DGMM_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                   \
    sycl::event dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,            \
                           const TYPE *a, int64_t lda, int64_t stridea, const TYPE *x,           \
                           int64_t incx, int64_t stridex, TYPE *c, int64_t ldc, int64_t stridec, \
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {   \
        return dgmm_batch(ROCBLAS_ROUTINE, queue, left_right, m, n, a, lda, stridea, x, incx,    \
                          stridex, c, ldc, stridec, batch_size, dependencies);                   \
    }

DGMM_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_sdgmm_strided_batched)
DGMM_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_ddgmm_strided_batched)
DGMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cdgmm_strided_batched)
DGMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zdgmm_strided_batched)

#undef DGMM_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event dgmm_batch(Func func, sycl::queue &queue, side *left_right, int64_t *m,
                              int64_t *n, const T **a, int64_t *lda, const T **x, int64_t *incx,
                              T **c, int64_t *ldc, int64_t group_count, int64_t *group_size,
                              const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(m[i], n[i], lda[i], ldc[i], incx[i], group_size[i]);
    }

    auto done = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            int64_t offset = 0;
            rocblas_status err;

            for (int64_t i = 0; i < group_count; i++) {
                auto **a_ = reinterpret_cast<const rocDataType **>(a);
                auto **x_ = reinterpret_cast<const rocDataType **>(x);
                auto **c_ = reinterpret_cast<rocDataType **>(c);
                ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_side_mode(left_right[i]),
                                        (int)m[i], (int)n[i], a_ + offset, (int)lda[i], x_ + offset,
                                        (int)incx[i], c_ + offset, (int)ldc[i], (int)group_size[i]);
                offset += group_size[i];
            }
        });
    });

    return done;
}

#define DGMM_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                            \
    sycl::event dgmm_batch(sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,          \
                           const TYPE **a, int64_t *lda, const TYPE **x, int64_t *incx, TYPE **c, \
                           int64_t *ldc, int64_t group_count, int64_t *group_size,                \
                           const std::vector<sycl::event> &dependencies) {                        \
        return dgmm_batch(ROCBLAS_ROUTINE, queue, left_right, m, n, a, lda, x, incx, c, ldc,      \
                          group_count, group_size, dependencies);                                 \
    }

DGMM_BATCH_LAUNCHER_USM(float, rocblas_sdgmm_batched)
DGMM_BATCH_LAUNCHER_USM(double, rocblas_ddgmm_batched)
DGMM_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cdgmm_batched)
DGMM_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zdgmm_batched)

#undef DGMM_BATCH_LAUNCHER

template <typename Func, typename T>
inline sycl::event gemm_batch(Func func, sycl::queue &queue, transpose transa, transpose transb,
                              int64_t m, int64_t n, int64_t k, T alpha, const T *a, int64_t lda,
                              int64_t stridea, const T *b, int64_t ldb, int64_t strideb, T beta,
                              T *c, int64_t ldc, int64_t stridec, int64_t batch_size,
                              const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, ldb, ldc, stridea, strideb, stridec, batch_size);

    auto done = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto b_ = reinterpret_cast<const rocDataType *>(b);
            auto c_ = reinterpret_cast<rocDataType *>(c);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_operation(transa),
                                    get_rocblas_operation(transb), m, n, k, (rocDataType *)&alpha,
                                    a_, lda, stridea, b_, ldb, strideb, (rocDataType *)&beta, c_,
                                    ldc, stridec, batch_size);
        });
    });

    return done;
}

#define GEMM_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                     \
    sycl::event gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,      \
                           int64_t n, int64_t k, TYPE alpha, const TYPE *a, int64_t lda,           \
                           int64_t stridea, const TYPE *b, int64_t ldb, int64_t strideb,           \
                           TYPE beta, TYPE *c, int64_t ldc, int64_t stridec, int64_t batch_size,   \
                           const std::vector<sycl::event> &dependencies) {                         \
        return gemm_batch(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, stridea, \
                          b, ldb, strideb, beta, c, ldc, stridec, batch_size, dependencies);       \
    }

GEMM_STRIDED_BATCH_LAUNCHER_USM(sycl::half, rocblas_hgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_sgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_dgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zgemm_strided_batched)

#undef GEMM_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event gemm_batch(Func func, sycl::queue &queue, transpose *transa, transpose *transb,
                              int64_t *m, int64_t *n, int64_t *k, T *alpha, const T **a,
                              int64_t *lda, const T **b, int64_t *ldb, T *beta, T **c, int64_t *ldc,
                              int64_t group_count, int64_t *group_size,
                              const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(m[i], n[i], k[i], lda[i], ldb[i], ldc[i], group_size[i]);
    }

    auto done = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            int64_t offset = 0;
            rocblas_status err;
            for (int64_t i = 0; i < group_count; i++) {
                auto **a_ = reinterpret_cast<const rocDataType **>(a);
                auto **b_ = reinterpret_cast<const rocDataType **>(b);
                auto **c_ = reinterpret_cast<rocDataType **>(c);
                ROCBLAS_ERROR_FUNC_SYNC(
                    func, err, handle, get_rocblas_operation(transa[i]),
                    get_rocblas_operation(transb[i]), (int)m[i], (int)n[i], (int)k[i],
                    (rocDataType *)&alpha[i], a_ + offset, (int)lda[i], b_ + offset, (int)ldb[i],
                    (rocDataType *)&beta[i], c_ + offset, (int)ldc[i], (int)group_size[i]);
                offset += group_size[i];
            }
        });
    });

    return done;
}

#define GEMM_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                            \
    sycl::event gemm_batch(sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,  \
                           int64_t *n, int64_t *k, TYPE *alpha, const TYPE **a, int64_t *lda,     \
                           const TYPE **b, int64_t *ldb, TYPE *beta, TYPE **c, int64_t *ldc,      \
                           int64_t group_count, int64_t *group_size,                              \
                           const std::vector<sycl::event> &dependencies) {                        \
        return gemm_batch(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, \
                          beta, c, ldc, group_count, group_size, dependencies);                   \
    }

GEMM_BATCH_LAUNCHER_USM(sycl::half, rocblas_hgemm_batched)
GEMM_BATCH_LAUNCHER_USM(float, rocblas_sgemm_batched)
GEMM_BATCH_LAUNCHER_USM(double, rocblas_dgemm_batched)
GEMM_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cgemm_batched)
GEMM_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zgemm_batched)

#undef GEMM_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event trsm_batch(Func func, sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, int64_t m, int64_t n, T alpha,
                              const T *a, int64_t lda, int64_t stridea, T *b, int64_t ldb,
                              int64_t strideb, int64_t batch_size,
                              const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, stridea, strideb, batch_size);

    auto done = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto b_ = reinterpret_cast<rocDataType *>(b);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_side_mode(left_right),
                                    get_rocblas_fill_mode(upper_lower),
                                    get_rocblas_operation(trans), get_rocblas_diag_type(unit_diag),
                                    m, n, (rocDataType *)&alpha, a_, lda, stridea, b_, ldb, strideb,
                                    batch_size);
        });
    });

    return done;
}

#define TRSM_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                     \
    sycl::event trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, \
                           diag unit_diag, int64_t m, int64_t n, TYPE alpha, const TYPE *a,        \
                           int64_t lda, int64_t stridea, TYPE *b, int64_t ldb, int64_t strideb,    \
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {     \
        return trsm_batch(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, \
                          alpha, a, lda, stridea, b, ldb, strideb, batch_size, dependencies);      \
    }

TRSM_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_strsm_strided_batched)
TRSM_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_dtrsm_strided_batched)
TRSM_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_ctrsm_strided_batched)
TRSM_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_ztrsm_strided_batched)

#undef TRSM_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event trsm_batch(Func func, sycl::queue &queue, side *left_right, uplo *upper_lower,
                              transpose *trans, diag *unit_diag, int64_t *m, int64_t *n, T *alpha,
                              const T **a, int64_t *lda, T **b, int64_t *ldb, int64_t group_count,
                              int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(m[i], n[i], lda[i], ldb[i], group_size[i]);
    }

    auto done = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            int64_t offset = 0;
            rocblas_status err;

            for (int64_t i = 0; i < group_count; i++) {
                auto **a_ = reinterpret_cast<const rocDataType **>(a);
                auto **b_ = reinterpret_cast<rocDataType **>(b);
                ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_side_mode(left_right[i]),
                                        get_rocblas_fill_mode(upper_lower[i]),
                                        get_rocblas_operation(trans[i]),
                                        get_rocblas_diag_type(unit_diag[i]), (int)m[i], (int)n[i],
                                        (rocDataType *)&alpha[i], a_ + offset, (int)lda[i],
                                        b_ + offset, (int)ldb[i], (int)group_size[i]);
                offset += group_size[i];
            }
        });
    });

    return done;
}

#define TRSM_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                             \
    sycl::event trsm_batch(sycl::queue &queue, side *left_right, uplo *upper_lower,                \
                           transpose *trans, diag *unit_diag, int64_t *m, int64_t *n, TYPE *alpha, \
                           const TYPE **a, int64_t *lda, TYPE **b, int64_t *ldb,                   \
                           int64_t group_count, int64_t *group_size,                               \
                           const std::vector<sycl::event> &dependencies) {                         \
        return trsm_batch(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, \
                          alpha, a, lda, b, ldb, group_count, group_size, dependencies);           \
    }

TRSM_BATCH_LAUNCHER_USM(float, rocblas_strsm_batched)
TRSM_BATCH_LAUNCHER_USM(double, rocblas_dtrsm_batched)
TRSM_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_ctrsm_batched)
TRSM_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_ztrsm_batched)

#undef TRSM_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event syrk_batch(Func func, sycl::queue &queue, uplo *upper_lower, transpose *trans,
                              int64_t *n, int64_t *k, T *alpha, const T **a, int64_t *lda, T *beta,
                              T **c, int64_t *ldc, int64_t group_count, int64_t *group_size,
                              const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(n[i], k[i], lda[i], ldc[i], group_size[i]);
    }

    auto done = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            int64_t offset = 0;
            rocblas_status err;

            for (int64_t i = 0; i < group_count; i++) {
                auto **a_ = reinterpret_cast<const rocDataType **>(a);
                auto **c_ = reinterpret_cast<rocDataType **>(c);
                ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower[i]),
                                        get_rocblas_operation(trans[i]), (int)n[i], (int)k[i],
                                        (rocDataType *)&alpha[i], a_ + offset, (int)lda[i],
                                        (rocDataType *)&beta[i], c_ + offset, (int)ldc[i],
                                        (int)group_size[i]);
                offset += group_size[i];
            }
        });
    });

    return done;
}

#define SYRK_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                           \
    sycl::event syrk_batch(sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,  \
                           int64_t *k, TYPE *alpha, const TYPE **a, int64_t *lda, TYPE *beta,    \
                           TYPE **c, int64_t *ldc, int64_t group_count, int64_t *group_size,     \
                           const std::vector<sycl::event> &dependencies) {                       \
        return syrk_batch(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, \
                          c, ldc, group_count, group_size, dependencies);                        \
    }

SYRK_BATCH_LAUNCHER_USM(float, rocblas_ssyrk_batched)
SYRK_BATCH_LAUNCHER_USM(double, rocblas_dsyrk_batched)
SYRK_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_csyrk_batched)
SYRK_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zsyrk_batched)

#undef SYRK_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event syrk_batch(Func func, sycl::queue &queue, uplo upper_lower, transpose trans,
                              int64_t n, int64_t k, const T alpha, const T *a, int64_t lda,
                              int64_t stridea, const T beta, T *c, int64_t ldc, int64_t stridec,
                              int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, k, lda, ldc, stridea, stridec, batch_size);

    auto done = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto c_ = reinterpret_cast<rocDataType *>(c);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                    get_rocblas_operation(trans), n, k, (rocDataType *)&alpha, a_,
                                    lda, stridea, (rocDataType *)&beta, c_, ldc, stridec,
                                    batch_size);
        });
    });

    return done;
}

#define SYRK_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                               \
    sycl::event syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, \
                           int64_t k, const TYPE alpha, const TYPE *a, int64_t lda,          \
                           int64_t stridea, const TYPE beta, TYPE *c, int64_t ldc,           \
                           int64_t stridec, int64_t batch_size,                              \
                           const std::vector<sycl::event> &dependencies) {                   \
        return syrk_batch(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda,   \
                          stridea, beta, c, ldc, stridec, batch_size, dependencies);         \
    }

SYRK_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_ssyrk_strided_batched)
SYRK_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_dsyrk_strided_batched)
SYRK_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_csyrk_strided_batched)
SYRK_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zsyrk_strided_batched)

#undef SYRK_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event omatcopy_batch(Func func, sycl::queue &queue, transpose trans, int64_t m,
                                  int64_t n, const T alpha, const T *a, int64_t lda,
                                  int64_t stridea, T *b, int64_t ldb, int64_t strideb,
                                  int64_t batch_size,
                                  const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, stridea, strideb, batch_size);

    const T beta = 0;
    const int64_t new_m = trans == oneapi::mkl::transpose::nontrans ? m : n;
    const int64_t new_n = trans == oneapi::mkl::transpose::nontrans ? n : m;

    auto done = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto b_ = reinterpret_cast<rocDataType *>(b);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_operation(trans),
                                    get_rocblas_operation(trans), new_m, new_n,
                                    (rocDataType *)&alpha, a_, lda, stridea, (rocDataType *)&beta,
                                    nullptr, lda, stridea, b_, ldb, strideb, batch_size);
        });
    });

    return done;
}

#define OMATCOPY_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                 \
    sycl::event omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,          \
                               const TYPE alpha, const TYPE *a, int64_t lda, int64_t stridea,      \
                               TYPE *b, int64_t ldb, int64_t strideb, int64_t batch_size,          \
                               const std::vector<sycl::event> &dependencies) {                     \
        return omatcopy_batch(ROCBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, stridea, b, ldb, \
                              strideb, batch_size, dependencies);                                  \
    }

OMATCOPY_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_sgeam_strided_batched)
OMATCOPY_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_dgeam_strided_batched)
OMATCOPY_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cgeam_strided_batched)
OMATCOPY_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zgeam_strided_batched)

#undef OMATCOPY_STRIDED_BATCH_LAUNCHER_USM

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                           float *ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "column_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                           double *ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "column_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                           std::complex<float> alpha, std::complex<float> *ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "column_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                           std::complex<double> alpha, std::complex<double> *ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "column_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

template <typename Func, typename T>
inline sycl::event omatadd_batch(Func func, sycl::queue &queue, transpose transa, transpose transb,
                                 int64_t m, int64_t n, const T alpha, const T *a, int64_t lda,
                                 int64_t stridea, const T beta, const T *b, int64_t ldb,
                                 int64_t strideb, T *c, int64_t ldc, int64_t stridec,
                                 int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, ldc, stridea, strideb, stridec, batch_size);

    auto done = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType *>(a);
            auto b_ = reinterpret_cast<const rocDataType *>(b);
            auto c_ = reinterpret_cast<rocDataType *>(c);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_operation(transa),
                                    get_rocblas_operation(transb), m, n, (rocDataType *)&alpha, a_,
                                    lda, stridea, (rocDataType *)&beta, b_, ldb, strideb, c_, ldc,
                                    stridec, batch_size);
        });
    });

    return done;
}

#define OMATADD_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                  \
    sycl::event omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,   \
                              int64_t n, const TYPE alpha, const TYPE *a, int64_t lda,             \
                              int64_t stridea, const TYPE beta, const TYPE *b, int64_t ldb,        \
                              int64_t strideb, TYPE *c, int64_t ldc, int64_t stridec,              \
                              int64_t batch_size, const std::vector<sycl::event> &dependencies) {  \
        return omatadd_batch(ROCBLAS_ROUTINE, queue, transa, transb, m, n, alpha, a, lda, stridea, \
                             beta, b, ldb, strideb, c, ldc, stridec, batch_size, dependencies);    \
    }

OMATADD_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_sgeam_strided_batched)
OMATADD_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_dgeam_strided_batched)
OMATADD_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cgeam_strided_batched)
OMATADD_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zgeam_strided_batched)

#undef OMATADD_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event omatcopy_batch(Func func, sycl::queue &queue, transpose *trans, int64_t *m,
                                  int64_t *n, T *alpha, const T **a, int64_t *lda, T **b,
                                  int64_t *ldb, int64_t group_count, int64_t *group_size,
                                  const std::vector<sycl::event> &dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(m[i], n[i], lda[i], ldb[i], group_size[i]);
    }

    auto done = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            int64_t offset = 0;
            rocblas_status err;

            for (int64_t i = 0; i < group_count; i++) {
                auto **a_ = reinterpret_cast<const rocDataType **>(a);
                auto **b_ = reinterpret_cast<rocDataType **>(b);

                const T beta = 0;
                const auto new_m = trans[i] == oneapi::mkl::transpose::nontrans ? m[i] : n[i];
                const auto new_n = trans[i] == oneapi::mkl::transpose::nontrans ? n[i] : m[i];

                ROCBLAS_ERROR_FUNC_SYNC(func, err, handle, get_rocblas_operation(trans[i]),
                                        get_rocblas_operation(trans[i]), (int)new_m, (int)new_n,
                                        (rocDataType *)&alpha[i], a_ + offset, (int)lda[i],
                                        (rocDataType *)&beta, nullptr, (int)lda[i], b_ + offset,
                                        (int)ldb[i], (int)group_size[i]);
                offset += group_size[i];
            }
        });
    });

    return done;
}

#define OMATCOPY_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                        \
    sycl::event omatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,      \
                               TYPE *alpha, const TYPE **a, int64_t *lda, TYPE **b, int64_t *ldb, \
                               int64_t group_count, int64_t *group_size,                          \
                               const std::vector<sycl::event> &dependencies) {                    \
        return omatcopy_batch(ROCBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, b, ldb,         \
                              group_count, group_size, dependencies);                             \
    }

OMATCOPY_BATCH_LAUNCHER_USM(float, rocblas_sgeam_batched)
OMATCOPY_BATCH_LAUNCHER_USM(double, rocblas_dgeam_batched)
OMATCOPY_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cgeam_batched)
OMATCOPY_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zgeam_batched)

#undef OMATCOPY_BATCH_LAUNCHER_USM

sycl::event imatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           float *alpha, float **ab, int64_t *lda, int64_t *ldb,
                           int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "column_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           double *alpha, double **ab, int64_t *lda, int64_t *ldb,
                           int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "column_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           std::complex<float> *alpha, std::complex<float> **ab, int64_t *lda,
                           int64_t *ldb, int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "column_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           std::complex<double> *alpha, std::complex<double> **ab, int64_t *lda,
                           int64_t *ldb, int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "column_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

} // namespace column_major

namespace row_major {

// Buffer APIs

template <typename Func, typename T>
inline void copy_batch(Func func, sycl::queue &queue, int64_t n, sycl::buffer<T, 1> &x,
                       int64_t incx, int64_t stridex, sycl::buffer<T, 1> &y, int64_t incy,
                       int64_t stridey, int64_t batch_size) {
    column_major::copy_batch(func, queue, n, x, incx, stridex, y, incy, stridey, batch_size);
}

#define COPY_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                     \
    void copy_batch(sycl::queue &queue, int64_t n, sycl::buffer<TYPE, 1> &x, int64_t incx,     \
                    int64_t stridex, sycl::buffer<TYPE, 1> &y, int64_t incy, int64_t stridey,  \
                    int64_t batch_size) {                                                      \
        copy_batch(ROCBLAS_ROUTINE, queue, n, x, incx, stridex, y, incy, stridey, batch_size); \
    }

COPY_STRIDED_BATCH_LAUNCHER(float, rocblas_scopy_strided_batched)
COPY_STRIDED_BATCH_LAUNCHER(double, rocblas_dcopy_strided_batched)
COPY_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_ccopy_strided_batched)
COPY_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_zcopy_strided_batched)

#undef COPY_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void axpy_batch(Func func, sycl::queue &queue, int64_t n, T alpha, sycl::buffer<T, 1> &x,
                       int64_t incx, int64_t stridex, sycl::buffer<T, 1> &y, int64_t incy,
                       int64_t stridey, int64_t batch_size) {
    column_major::axpy_batch(func, queue, n, alpha, x, incx, stridex, y, incy, stridey, batch_size);
}

#define AXPY_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                 \
    void axpy_batch(sycl::queue &queue, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1> &x,   \
                    int64_t incx, int64_t stridex, sycl::buffer<TYPE, 1> &y, int64_t incy, \
                    int64_t stridey, int64_t batch_size) {                                 \
        axpy_batch(ROCBLAS_ROUTINE, queue, n, alpha, x, incx, stridex, y, incy, stridey,   \
                   batch_size);                                                            \
    }

AXPY_STRIDED_BATCH_LAUNCHER(float, rocblas_saxpy_strided_batched)
AXPY_STRIDED_BATCH_LAUNCHER(double, rocblas_daxpy_strided_batched)
AXPY_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_caxpy_strided_batched)
AXPY_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_zaxpy_strided_batched)

#undef AXPY_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void gemv_batch(Func func, sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                       std::complex<T> alpha, sycl::buffer<std::complex<T>, 1> &a, int64_t lda,
                       int64_t stridea, sycl::buffer<std::complex<T>, 1> &x, int64_t incx,
                       int64_t stridex, std::complex<T> beta, sycl::buffer<std::complex<T>, 1> &y,
                       int64_t incy, int64_t stridey, int64_t batch_size) {
    auto new_trans = trans == oneapi::mkl::transpose::nontrans ? oneapi::mkl::transpose::trans
                                                               : oneapi::mkl::transpose::nontrans;

    if (trans == oneapi::mkl::transpose::conjtrans) {
        alpha = std::conj(alpha);
        beta = std::conj(beta);

        if (m > 0) {
            queue.submit([&](sycl::handler &cgh) {
                const auto abs_incx = std::abs(incx);
                const auto abs_stridex = std::abs(stridex);

                auto acc = x.template get_access<sycl::access::mode::read_write>(cgh);
                cgh.parallel_for(
                    sycl::range{ (std::size_t)batch_size, (std::size_t)m }, [=](sycl::item<2> it) {
                        const auto index = it.get_id(0) * abs_stridex + it.get_id(1) * abs_incx;
                        acc[index] = std::conj(acc[index]);
                    });
            });

            if (n > 0) {
                queue.submit([&](sycl::handler &cgh) {
                    const auto abs_incy = std::abs(incy);
                    const auto abs_stridey = std::abs(stridey);

                    auto acc = y.template get_access<sycl::access::mode::read_write>(cgh);
                    cgh.parallel_for(sycl::range{ (std::size_t)batch_size, (std::size_t)n },
                                     [=](sycl::item<2> it) {
                                         const auto index =
                                             it.get_id(0) * abs_stridey + it.get_id(1) * abs_incy;
                                         acc[index] = std::conj(acc[index]);
                                     });
                });
            }
        }
    }

    column_major::gemv_batch(func, queue, new_trans, n, m, alpha, a, lda, stridea, x, incx, stridex,
                             beta, y, incy, stridey, batch_size);

    if (trans == oneapi::mkl::transpose::conjtrans) {
        if (n > 0) {
            queue.submit([&](sycl::handler &cgh) {
                const auto abs_incy = std::abs(incy);
                const auto abs_stridey = std::abs(stridey);

                auto acc = y.template get_access<sycl::access::mode::read_write>(cgh);
                cgh.parallel_for(
                    sycl::range{ (std::size_t)batch_size, (std::size_t)n }, [=](sycl::item<2> it) {
                        const auto index = it.get_id(0) * abs_stridey + it.get_id(1) * abs_incy;
                        acc[index] = std::conj(acc[index]);
                    });
            });
        }
    }
}

template <typename Func, typename T>
inline void gemv_batch(Func func, sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                       T alpha, sycl::buffer<T, 1> &a, int64_t lda, int64_t stridea,
                       sycl::buffer<T, 1> &x, int64_t incx, int64_t stridex, T beta,
                       sycl::buffer<T, 1> &y, int64_t incy, int64_t stridey, int64_t batch_size) {
    auto new_trans = trans == oneapi::mkl::transpose::nontrans ? oneapi::mkl::transpose::trans
                                                               : oneapi::mkl::transpose::nontrans;

    column_major::gemv_batch(func, queue, new_trans, n, m, alpha, a, lda, stridea, x, incx, stridex,
                             beta, y, incy, stridey, batch_size);
}

#define GEMV_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                         \
    void gemv_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, TYPE alpha,         \
                    sycl::buffer<TYPE, 1> &a, int64_t lda, int64_t stridea,                        \
                    sycl::buffer<TYPE, 1> &x, int64_t incx, int64_t stridex, TYPE beta,            \
                    sycl::buffer<TYPE, 1> &y, int64_t incy, int64_t stridey, int64_t batch_size) { \
        gemv_batch(ROCBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, stridea, x, incx, stridex,  \
                   beta, y, incy, stridey, batch_size);                                            \
    }

GEMV_STRIDED_BATCH_LAUNCHER(float, rocblas_sgemv_strided_batched)
GEMV_STRIDED_BATCH_LAUNCHER(double, rocblas_dgemv_strided_batched)
GEMV_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_cgemv_strided_batched)
GEMV_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_zgemv_strided_batched)

#undef GEMV_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void dgmm_batch(Func func, sycl::queue &queue, side left_right, int64_t m, int64_t n,
                       sycl::buffer<T, 1> &a, int64_t lda, int64_t stridea, sycl::buffer<T, 1> &x,
                       int64_t incx, int64_t stridex, sycl::buffer<T, 1> &c, int64_t ldc,
                       int64_t stridec, int64_t batch_size) {
    auto new_side =
        left_right == oneapi::mkl::side::left ? oneapi::mkl::side::right : oneapi::mkl::side::left;

    column_major::dgmm_batch(func, queue, new_side, n, m, a, lda, stridea, x, incx, stridex, c, ldc,
                             stridec, batch_size);
}

#define DGMM_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                         \
    void dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,                     \
                    sycl::buffer<TYPE, 1> &a, int64_t lda, int64_t stridea,                        \
                    sycl::buffer<TYPE, 1> &x, int64_t incx, int64_t stridex,                       \
                    sycl::buffer<TYPE, 1> &c, int64_t ldc, int64_t stridec, int64_t batch_size) {  \
        dgmm_batch(ROCBLAS_ROUTINE, queue, left_right, m, n, a, lda, stridea, x, incx, stridex, c, \
                   ldc, stridec, batch_size);                                                      \
    }

DGMM_STRIDED_BATCH_LAUNCHER(float, rocblas_sdgmm_strided_batched)
DGMM_STRIDED_BATCH_LAUNCHER(double, rocblas_ddgmm_strided_batched)
DGMM_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_cdgmm_strided_batched)
DGMM_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_zdgmm_strided_batched)

#undef DGMM_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void gemm_batch(Func func, sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                       int64_t n, int64_t k, T alpha, sycl::buffer<T, 1> &a, int64_t lda,
                       int64_t stridea, sycl::buffer<T, 1> &b, int64_t ldb, int64_t strideb, T beta,
                       sycl::buffer<T, 1> &c, int64_t ldc, int64_t stridec, int64_t batch_size) {
    auto new_transa = transb;
    auto new_transb = transa;

    column_major::gemm_batch(func, queue, new_transa, new_transb, n, m, k, alpha, b, ldb, strideb,
                             a, lda, stridea, beta, c, ldc, stridec, batch_size);
}

#define GEMM_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                         \
    void gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,  \
                    int64_t k, TYPE alpha, sycl::buffer<TYPE, 1> &a, int64_t lda, int64_t stridea, \
                    sycl::buffer<TYPE, 1> &b, int64_t ldb, int64_t strideb, TYPE beta,             \
                    sycl::buffer<TYPE, 1> &c, int64_t ldc, int64_t stridec, int64_t batch_size) {  \
        gemm_batch(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, stridea, b,     \
                   ldb, strideb, beta, c, ldc, stridec, batch_size);                               \
    }

GEMM_STRIDED_BATCH_LAUNCHER(sycl::half, rocblas_hgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER(float, rocblas_sgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER(double, rocblas_dgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_cgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_zgemm_strided_batched)

#undef GEMM_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void trsm_batch(Func func, sycl::queue &queue, side left_right, uplo upper_lower,
                       transpose trans, diag unit_diag, int64_t m, int64_t n, T alpha,
                       sycl::buffer<T, 1> &a, int64_t lda, int64_t stridea, sycl::buffer<T, 1> &b,
                       int64_t ldb, int64_t strideb, int64_t batch_size) {
    auto new_side =
        left_right == oneapi::mkl::side::left ? oneapi::mkl::side::right : oneapi::mkl::side::left;
    auto new_uplo = upper_lower == oneapi::mkl::uplo::lower ? oneapi::mkl::uplo::upper
                                                            : oneapi::mkl::uplo::lower;

    column_major::trsm_batch(func, queue, new_side, new_uplo, trans, unit_diag, n, m, alpha, a, lda,
                             stridea, b, ldb, strideb, batch_size);
}

#define TRSM_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                         \
    void trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,        \
                    diag unit_diag, int64_t m, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1> &a,    \
                    int64_t lda, int64_t stridea, sycl::buffer<TYPE, 1> &b, int64_t ldb,           \
                    int64_t strideb, int64_t batch_size) {                                         \
        trsm_batch(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, \
                   a, lda, stridea, b, ldb, strideb, batch_size);                                  \
    }

TRSM_STRIDED_BATCH_LAUNCHER(float, rocblas_strsm_strided_batched)
TRSM_STRIDED_BATCH_LAUNCHER(double, rocblas_dtrsm_strided_batched)
TRSM_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_ctrsm_strided_batched)
TRSM_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_ztrsm_strided_batched)

#undef TRSM_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void syrk_batch(Func func, sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                       int64_t k, T alpha, sycl::buffer<T, 1> &a, int64_t lda, int64_t stridea,
                       T beta, sycl::buffer<T, 1> &c, int64_t ldc, int64_t stridec,
                       int64_t batch_size) {
    auto new_uplo = upper_lower == oneapi::mkl::uplo::lower ? oneapi::mkl::uplo::upper
                                                            : oneapi::mkl::uplo::lower;
    auto new_trans = trans == oneapi::mkl::transpose::nontrans ? oneapi::mkl::transpose::trans
                                                               : oneapi::mkl::transpose::nontrans;

    column_major::syrk_batch(func, queue, new_uplo, new_trans, n, k, alpha, a, lda, stridea, beta,
                             c, ldc, stridec, batch_size);
}

#define SYRK_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                         \
    void syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,   \
                    TYPE alpha, sycl::buffer<TYPE, 1> &a, int64_t lda, int64_t stridea, TYPE beta, \
                    sycl::buffer<TYPE, 1> &c, int64_t ldc, int64_t stridec, int64_t batch_size) {  \
        syrk_batch(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, stridea, beta, \
                   c, ldc, stridec, batch_size);                                                   \
    }

SYRK_STRIDED_BATCH_LAUNCHER(float, rocblas_ssyrk_strided_batched)
SYRK_STRIDED_BATCH_LAUNCHER(double, rocblas_dsyrk_strided_batched)
SYRK_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_csyrk_strided_batched)
SYRK_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_zsyrk_strided_batched)

#undef SYRK_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void omatcopy_batch(Func func, sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                           const T alpha, sycl::buffer<T, 1> &a, int64_t lda, int64_t stridea,
                           sycl::buffer<T, 1> &b, int64_t ldb, int64_t strideb,
                           int64_t batch_size) {
    return column_major::omatcopy_batch(func, queue, trans, n, m, alpha, a, lda, stridea, b, ldb,
                                        strideb, batch_size);
}

#define OMATCOPY_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                    \
    void omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,                \
                        const TYPE alpha, sycl::buffer<TYPE, 1> &a, int64_t lda, int64_t stridea, \
                        sycl::buffer<TYPE, 1> &b, int64_t ldb, int64_t strideb,                   \
                        int64_t batch_size) {                                                     \
        omatcopy_batch(ROCBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, stridea, b, ldb,       \
                       strideb, batch_size);                                                      \
    }

OMATCOPY_STRIDED_BATCH_LAUNCHER(float, rocblas_sgeam_strided_batched)
OMATCOPY_STRIDED_BATCH_LAUNCHER(double, rocblas_dgeam_strided_batched)
OMATCOPY_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_cgeam_strided_batched)
OMATCOPY_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_zgeam_strided_batched)

#undef OMATCOPY_STRIDED_BATCH_LAUNCHER

void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                    sycl::buffer<float, 1> &ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size) {
    throw unimplemented("blas", "row_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                    sycl::buffer<double, 1> &ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size) {
    throw unimplemented("blas", "row_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
    throw unimplemented("blas", "row_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
    throw unimplemented("blas", "row_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

template <typename Func, typename T>
inline void omatadd_batch(Func func, sycl::queue &queue, transpose transa, transpose transb,
                          int64_t m, int64_t n, const T alpha, sycl::buffer<T, 1> &a, int64_t lda,
                          int64_t stridea, const T beta, sycl::buffer<T, 1> &b, int64_t ldb,
                          int64_t strideb, sycl::buffer<T, 1> &c, int64_t ldc, int64_t stridec,
                          int64_t batch_size) {
    return column_major::omatadd_batch(func, queue, transa, transb, n, m, alpha, a, lda, stridea,
                                       beta, b, ldb, strideb, c, ldc, stridec, batch_size);
}

#define OMATADD_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                     \
    void omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,         \
                       int64_t n, const TYPE alpha, sycl::buffer<TYPE, 1> &a, int64_t lda,        \
                       int64_t stridea, const TYPE beta, sycl::buffer<TYPE, 1> &b, int64_t ldb,   \
                       int64_t strideb, sycl::buffer<TYPE, 1> &c, int64_t ldc, int64_t stridec,   \
                       int64_t batch_size) {                                                      \
        omatadd_batch(ROCBLAS_ROUTINE, queue, transa, transb, m, n, alpha, a, lda, stridea, beta, \
                      b, ldb, strideb, c, ldc, stridec, batch_size);                              \
    }

OMATADD_STRIDED_BATCH_LAUNCHER(float, rocblas_sgeam_strided_batched)
OMATADD_STRIDED_BATCH_LAUNCHER(double, rocblas_dgeam_strided_batched)
OMATADD_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_cgeam_strided_batched)
OMATADD_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_zgeam_strided_batched)

#undef OMATADD_STRIDED_BATCH_LAUNCHER

// USM APIs

template <typename Func, typename T>
inline sycl::event copy_batch(Func func, sycl::queue &queue, int64_t *n, const T **x, int64_t *incx,
                              T **y, int64_t *incy, int64_t group_count, int64_t *group_size,
                              const std::vector<sycl::event> &dependencies) {
    return column_major::copy_batch(func, queue, n, x, incx, y, incy, group_count, group_size,
                                    dependencies);
}

#define COPY_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                          \
    sycl::event copy_batch(sycl::queue &queue, int64_t *n, const TYPE **x, int64_t *incx,       \
                           TYPE **y, int64_t *incy, int64_t group_count, int64_t *group_size,   \
                           const std::vector<sycl::event> &dependencies) {                      \
        return copy_batch(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy, group_count, group_size, \
                          dependencies);                                                        \
    }

COPY_BATCH_LAUNCHER_USM(float, rocblas_scopy_batched)
COPY_BATCH_LAUNCHER_USM(double, rocblas_dcopy_batched)
COPY_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_ccopy_batched)
COPY_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zcopy_batched)

#undef COPY_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event copy_batch(Func func, sycl::queue &queue, int64_t n, const T *x, int64_t incx,
                              int64_t stridex, T *y, int64_t incy, int64_t stridey,
                              int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    return column_major::copy_batch(func, queue, n, x, incx, stridex, y, incy, stridey, batch_size,
                                    dependencies);
}

#define COPY_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                 \
    sycl::event copy_batch(sycl::queue &queue, int64_t n, const TYPE *x, int64_t incx,         \
                           int64_t stridex, TYPE *y, int64_t incy, int64_t stridey,            \
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) { \
        return copy_batch(ROCBLAS_ROUTINE, queue, n, x, incx, stridex, y, incy, stridey,       \
                          batch_size, dependencies);                                           \
    }

COPY_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_scopy_strided_batched)
COPY_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_dcopy_strided_batched)
COPY_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_ccopy_strided_batched)
COPY_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zcopy_strided_batched)

#undef COPY_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event axpy_batch(Func func, sycl::queue &queue, int64_t *n, T *alpha, const T **x,
                              int64_t *incx, T **y, int64_t *incy, int64_t group_count,
                              int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    return column_major::axpy_batch(func, queue, n, alpha, x, incx, y, incy, group_count,
                                    group_size, dependencies);
}

#define AXPY_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                          \
    sycl::event axpy_batch(sycl::queue &queue, int64_t *n, TYPE *alpha, const TYPE **x,         \
                           int64_t *incx, TYPE **y, int64_t *incy, int64_t group_count,         \
                           int64_t *group_size, const std::vector<sycl::event> &dependencies) { \
        return axpy_batch(ROCBLAS_ROUTINE, queue, n, alpha, x, incx, y, incy, group_count,      \
                          group_size, dependencies);                                            \
    }

AXPY_BATCH_LAUNCHER_USM(float, rocblas_saxpy_batched)
AXPY_BATCH_LAUNCHER_USM(double, rocblas_daxpy_batched)
AXPY_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_caxpy_batched)
AXPY_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zaxpy_batched)

#undef AXPY_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event axpy_batch(Func func, sycl::queue &queue, int64_t n, T alpha, const T *x,
                              int64_t incx, int64_t stridex, T *y, int64_t incy, int64_t stridey,
                              int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    return column_major::axpy_batch(func, queue, n, alpha, x, incx, stridex, y, incy, stridey,
                                    batch_size, dependencies);
}

#define AXPY_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                     \
    sycl::event axpy_batch(sycl::queue &queue, int64_t n, TYPE alpha, const TYPE *x, int64_t incx, \
                           int64_t stridex, TYPE *y, int64_t incy, int64_t stridey,                \
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {     \
        return axpy_batch(ROCBLAS_ROUTINE, queue, n, alpha, x, incx, stridex, y, incy, stridey,    \
                          batch_size, dependencies);                                               \
    }

AXPY_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_saxpy_strided_batched)
AXPY_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_daxpy_strided_batched)
AXPY_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_caxpy_strided_batched)
AXPY_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zaxpy_strided_batched)

#undef AXPY_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event gemv_batch(Func func, sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                              std::complex<T> alpha, const std::complex<T> *a, int64_t lda,
                              int64_t stridea, const std::complex<T> *x, int64_t incx,
                              int64_t stridex, std::complex<T> beta, std::complex<T> *y,
                              int64_t incy, int64_t stridey, int64_t batch_size,
                              const std::vector<sycl::event> &dependencies) {
    sycl::event done;

    auto new_trans = trans == oneapi::mkl::transpose::nontrans ? oneapi::mkl::transpose::trans
                                                               : oneapi::mkl::transpose::nontrans;

    if (trans == oneapi::mkl::transpose::conjtrans) {
        alpha = std::conj(alpha);
        beta = std::conj(beta);

        if (m > 0) {
            done = queue.submit([&](sycl::handler &cgh) {
                const auto abs_incx = std::abs(incx);
                const auto abs_stridex = std::abs(stridex);

                auto acc = (std::complex<T> *)x;
                cgh.parallel_for(
                    sycl::range{ (std::size_t)batch_size, (std::size_t)m }, [=](sycl::item<2> it) {
                        const auto index = it.get_id(0) * abs_stridex + it.get_id(1) * abs_incx;
                        acc[index] = std::conj(acc[index]);
                    });
            });

            if (n > 0) {
                done = queue.submit([&](sycl::handler &cgh) {
                    const auto abs_incy = std::abs(incy);
                    const auto abs_stridey = std::abs(stridey);

                    cgh.parallel_for(sycl::range{ (std::size_t)batch_size, (std::size_t)n },
                                     [=](sycl::item<2> it) {
                                         const auto index =
                                             it.get_id(0) * abs_stridey + it.get_id(1) * abs_incy;
                                         y[index] = std::conj(y[index]);
                                     });
                });
            }
        }
    }

    done.wait_and_throw();

    done = column_major::gemv_batch(func, queue, new_trans, n, m, alpha, a, lda, stridea, x, incx,
                                    stridex, beta, y, incy, stridey, batch_size, dependencies);

    if (trans == oneapi::mkl::transpose::conjtrans) {
        if (n > 0) {
            done = queue.submit([&](sycl::handler &cgh) {
                const auto abs_incy = std::abs(incy);
                const auto abs_stridey = std::abs(stridey);

                cgh.depends_on(done);
                cgh.parallel_for(
                    sycl::range{ (std::size_t)batch_size, (std::size_t)n }, [=](sycl::item<2> it) {
                        const auto index = it.get_id(0) * abs_stridey + it.get_id(1) * abs_incy;
                        y[index] = std::conj(y[index]);
                    });
            });
        }
    }

    return done;
}

template <typename Func, typename T>
inline sycl::event gemv_batch(Func func, sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                              T alpha, const T *a, int64_t lda, int64_t stridea, const T *x,
                              int64_t incx, int64_t stridex, T beta, T *y, int64_t incy,
                              int64_t stridey, int64_t batch_size,
                              const std::vector<sycl::event> &dependencies) {
    auto new_trans = trans == oneapi::mkl::transpose::nontrans ? oneapi::mkl::transpose::trans
                                                               : oneapi::mkl::transpose::nontrans;

    return column_major::gemv_batch(func, queue, new_trans, n, m, alpha, a, lda, stridea, x, incx,
                                    stridex, beta, y, incy, stridey, batch_size, dependencies);
}

#define GEMV_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                    \
    sycl::event gemv_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, TYPE alpha, \
                           const TYPE *a, int64_t lda, int64_t stridea, const TYPE *x,            \
                           int64_t incx, int64_t stridex, TYPE beta, TYPE *y, int64_t incy,       \
                           int64_t stridey, int64_t batch_size,                                   \
                           const std::vector<sycl::event> &dependencies) {                        \
        return gemv_batch(ROCBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, stridea, x, incx,   \
                          stridex, beta, y, incy, stridey, batch_size, dependencies);             \
    }

GEMV_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_sgemv_strided_batched)
GEMV_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_dgemv_strided_batched)
GEMV_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cgemv_strided_batched)
GEMV_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zgemv_strided_batched)

#undef GEMV_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event gemv_batch(Func func, sycl::queue &queue, transpose *trans, int64_t *m,
                              int64_t *n, std::complex<T> *alpha, const std::complex<T> **a,
                              int64_t *lda, const std::complex<T> **x, int64_t *incx,
                              std::complex<T> *beta, std::complex<T> **y, int64_t *incy,
                              int64_t group_count, int64_t *group_size,
                              const std::vector<sycl::event> &dependencies) {
    sycl::event done;

    int64_t stride = 0;
    for (int64_t i = 0; i < group_count; i++) {
        if (trans[i] == oneapi::mkl::transpose::conjtrans) {
            alpha[i] = std::conj(alpha[i]);
            beta[i] = std::conj(beta[i]);

            if (m[i] > 0) {
                done = queue.submit([&](sycl::handler &cgh) {
                    const auto abs_incx = std::abs(incx[i]);

                    auto acc = (std::complex<T> **)x;
                    cgh.parallel_for(sycl::range{ (std::size_t)group_size[i], (std::size_t)m[i] },
                                     [=](sycl::item<2> it) {
                                         const auto col = it.get_id(0) + stride;
                                         const auto row = it.get_id(1) * abs_incx;
                                         acc[col][row] = std::conj(acc[col][row]);
                                     });
                });

                if (n[i] > 0) {
                    done = queue.submit([&](sycl::handler &cgh) {
                        const auto abs_incy = std::abs(incy[i]);

                        cgh.parallel_for(
                            sycl::range{ (std::size_t)group_size[i], (std::size_t)n[i] },
                            [=](sycl::item<2> it) {
                                const auto col = it.get_id(0) + stride;
                                const auto row = it.get_id(1) * abs_incy;
                                y[col][row] = std::conj(y[col][row]);
                            });
                    });
                }
            }
        }
        stride += group_size[i];
    }

    done.wait_and_throw();

    auto tmp_trans = std::vector<transpose>{ (std::size_t)group_count };
    for (int64_t i = 0; i < group_count; i++) {
        const auto new_trans = trans[i] == oneapi::mkl::transpose::nontrans
                                   ? oneapi::mkl::transpose::trans
                                   : oneapi::mkl::transpose::nontrans;
        tmp_trans[i] = trans[i];
        trans[i] = new_trans;
    }
    done = column_major::gemv_batch(func, queue, trans, n, m, alpha, a, lda, x, incx, beta, y, incy,
                                    group_count, group_size, dependencies);
    done.wait_and_throw();
    for (int64_t i = 0; i < group_count; i++) {
        trans[i] = tmp_trans[i];
    }

    stride = 0;
    for (int64_t i = 0; i < group_count; i++) {
        if (trans[i] == oneapi::mkl::transpose::conjtrans) {
            if (n[i] > 0) {
                done = queue.submit([&](sycl::handler &cgh) {
                    const auto abs_incy = std::abs(incy[i]);

                    cgh.parallel_for(sycl::range{ (std::size_t)group_size[i], (std::size_t)n[i] },
                                     [=](sycl::item<2> it) {
                                         const auto col = it.get_id(0) + stride;
                                         const auto row = it.get_id(1) * abs_incy;
                                         y[col][row] = std::conj(y[col][row]);
                                     });
                });
            }
        }
        stride += group_size[i];
    }

    return done;
}

template <typename Func, typename T>
inline sycl::event gemv_batch(Func func, sycl::queue &queue, transpose *trans, int64_t *m,
                              int64_t *n, T *alpha, const T **a, int64_t *lda, const T **x,
                              int64_t *incx, T *beta, T **y, int64_t *incy, int64_t group_count,
                              int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    auto tmp_trans = std::vector<transpose>{ static_cast<std::size_t>(group_count) };

    for (int64_t i = 0; i < group_count; i++) {
        const auto new_trans = trans[i] == oneapi::mkl::transpose::nontrans
                                   ? oneapi::mkl::transpose::trans
                                   : oneapi::mkl::transpose::nontrans;
        tmp_trans[i] = trans[i];
        trans[i] = new_trans;
    }
    auto done = column_major::gemv_batch(func, queue, trans, n, m, alpha, a, lda, x, incx, beta, y,
                                         incy, group_count, group_size, dependencies);
    done.wait_and_throw();
    for (int64_t i = 0; i < group_count; i++) {
        trans[i] = tmp_trans[i];
    }

    return done;
}

#define GEMV_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                             \
    sycl::event gemv_batch(                                                                        \
        sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n, TYPE *alpha, const TYPE **a, \
        int64_t *lda, const TYPE **x, int64_t *incx, TYPE *beta, TYPE **y, int64_t *incy,          \
        int64_t group_count, int64_t *group_size, const std::vector<sycl::event> &dependencies) {  \
        return gemv_batch(ROCBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, x, incx, beta, y,    \
                          incy, group_count, group_size, dependencies);                            \
    }

GEMV_BATCH_LAUNCHER_USM(float, rocblas_sgemv_batched)
GEMV_BATCH_LAUNCHER_USM(double, rocblas_dgemv_batched)
GEMV_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cgemv_batched)
GEMV_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zgemv_batched)

#undef GEMV_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event dgmm_batch(Func func, sycl::queue &queue, side left_right, int64_t m, int64_t n,
                              const T *a, int64_t lda, int64_t stridea, const T *x, int64_t incx,
                              int64_t stridex, T *c, int64_t ldc, int64_t stridec,
                              int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    auto new_side =
        left_right == oneapi::mkl::side::left ? oneapi::mkl::side::right : oneapi::mkl::side::left;

    return column_major::dgmm_batch(func, queue, new_side, n, m, a, lda, stridea, x, incx, stridex,
                                    c, ldc, stridec, batch_size, dependencies);
}

#define DGMM_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                   \
    sycl::event dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,            \
                           const TYPE *a, int64_t lda, int64_t stridea, const TYPE *x,           \
                           int64_t incx, int64_t stridex, TYPE *c, int64_t ldc, int64_t stridec, \
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {   \
        return dgmm_batch(ROCBLAS_ROUTINE, queue, left_right, m, n, a, lda, stridea, x, incx,    \
                          stridex, c, ldc, stridec, batch_size, dependencies);                   \
    }

DGMM_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_sdgmm_strided_batched)
DGMM_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_ddgmm_strided_batched)
DGMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cdgmm_strided_batched)
DGMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zdgmm_strided_batched)

#undef DGMM_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event dgmm_batch(Func func, sycl::queue &queue, side *left_right, int64_t *m,
                              int64_t *n, const T **a, int64_t *lda, const T **x, int64_t *incx,
                              T **c, int64_t *ldc, int64_t group_count, int64_t *group_size,
                              const std::vector<sycl::event> &dependencies) {
    for (int64_t i = 0; i < group_count; i++) {
        const auto new_side = left_right[i] == oneapi::mkl::side::left ? oneapi::mkl::side::right
                                                                       : oneapi::mkl::side::left;
        left_right[i] = new_side;
    }

    return column_major::dgmm_batch(func, queue, left_right, n, m, a, lda, x, incx, c, ldc,
                                    group_count, group_size, dependencies);
}

#define DGMM_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                            \
    sycl::event dgmm_batch(sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,          \
                           const TYPE **a, int64_t *lda, const TYPE **x, int64_t *incx, TYPE **c, \
                           int64_t *ldc, int64_t group_count, int64_t *group_size,                \
                           const std::vector<sycl::event> &dependencies) {                        \
        return dgmm_batch(ROCBLAS_ROUTINE, queue, left_right, m, n, a, lda, x, incx, c, ldc,      \
                          group_count, group_size, dependencies);                                 \
    }

DGMM_BATCH_LAUNCHER_USM(float, rocblas_sdgmm_batched)
DGMM_BATCH_LAUNCHER_USM(double, rocblas_ddgmm_batched)
DGMM_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cdgmm_batched)
DGMM_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zdgmm_batched)

#undef DGMM_BATCH_LAUNCHER

template <typename Func, typename T>
inline sycl::event gemm_batch(Func func, sycl::queue &queue, transpose transa, transpose transb,
                              int64_t m, int64_t n, int64_t k, T alpha, const T *a, int64_t lda,
                              int64_t stridea, const T *b, int64_t ldb, int64_t strideb, T beta,
                              T *c, int64_t ldc, int64_t stridec, int64_t batch_size,
                              const std::vector<sycl::event> &dependencies) {
    auto new_transa = transb;
    auto new_transb = transa;

    return column_major::gemm_batch(func, queue, new_transa, new_transb, n, m, k, alpha, b, ldb,
                                    strideb, a, lda, stridea, beta, c, ldc, stridec, batch_size,
                                    dependencies);
}

#define GEMM_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                     \
    sycl::event gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,      \
                           int64_t n, int64_t k, TYPE alpha, const TYPE *a, int64_t lda,           \
                           int64_t stridea, const TYPE *b, int64_t ldb, int64_t strideb,           \
                           TYPE beta, TYPE *c, int64_t ldc, int64_t stridec, int64_t batch_size,   \
                           const std::vector<sycl::event> &dependencies) {                         \
        return gemm_batch(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, stridea, \
                          b, ldb, strideb, beta, c, ldc, stridec, batch_size, dependencies);       \
    }

GEMM_STRIDED_BATCH_LAUNCHER_USM(sycl::half, rocblas_hgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_sgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_dgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zgemm_strided_batched)

#undef GEMM_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event gemm_batch(Func func, sycl::queue &queue, transpose *transa, transpose *transb,
                              int64_t *m, int64_t *n, int64_t *k, T *alpha, const T **a,
                              int64_t *lda, const T **b, int64_t *ldb, T *beta, T **c, int64_t *ldc,
                              int64_t group_count, int64_t *group_size,
                              const std::vector<sycl::event> &dependencies) {
    for (int64_t i = 0; i < group_count; i++) {
        std::swap(transa[i], transb[i]);
    }

    return column_major::gemm_batch(func, queue, transa, transb, n, m, k, alpha, b, ldb, a, lda,
                                    beta, c, ldc, group_count, group_size, dependencies);
}

#define GEMM_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                            \
    sycl::event gemm_batch(sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,  \
                           int64_t *n, int64_t *k, TYPE *alpha, const TYPE **a, int64_t *lda,     \
                           const TYPE **b, int64_t *ldb, TYPE *beta, TYPE **c, int64_t *ldc,      \
                           int64_t group_count, int64_t *group_size,                              \
                           const std::vector<sycl::event> &dependencies) {                        \
        return gemm_batch(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, \
                          beta, c, ldc, group_count, group_size, dependencies);                   \
    }

GEMM_BATCH_LAUNCHER_USM(sycl::half, rocblas_hgemm_batched)
GEMM_BATCH_LAUNCHER_USM(float, rocblas_sgemm_batched)
GEMM_BATCH_LAUNCHER_USM(double, rocblas_dgemm_batched)
GEMM_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cgemm_batched)
GEMM_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zgemm_batched)

#undef GEMM_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event trsm_batch(Func func, sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, int64_t m, int64_t n, T alpha,
                              const T *a, int64_t lda, int64_t stridea, T *b, int64_t ldb,
                              int64_t strideb, int64_t batch_size,
                              const std::vector<sycl::event> &dependencies) {
    auto new_side =
        left_right == oneapi::mkl::side::left ? oneapi::mkl::side::right : oneapi::mkl::side::left;
    auto new_uplo = upper_lower == oneapi::mkl::uplo::lower ? oneapi::mkl::uplo::upper
                                                            : oneapi::mkl::uplo::lower;

    return column_major::trsm_batch(func, queue, new_side, new_uplo, trans, unit_diag, n, m, alpha,
                                    a, lda, stridea, b, ldb, strideb, batch_size, dependencies);
}

#define TRSM_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                     \
    sycl::event trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, \
                           diag unit_diag, int64_t m, int64_t n, TYPE alpha, const TYPE *a,        \
                           int64_t lda, int64_t stridea, TYPE *b, int64_t ldb, int64_t strideb,    \
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {     \
        return trsm_batch(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, \
                          alpha, a, lda, stridea, b, ldb, strideb, batch_size, dependencies);      \
    }

TRSM_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_strsm_strided_batched)
TRSM_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_dtrsm_strided_batched)
TRSM_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_ctrsm_strided_batched)
TRSM_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_ztrsm_strided_batched)

#undef TRSM_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event trsm_batch(Func func, sycl::queue &queue, side *left_right, uplo *upper_lower,
                              transpose *trans, diag *unit_diag, int64_t *m, int64_t *n, T *alpha,
                              const T **a, int64_t *lda, T **b, int64_t *ldb, int64_t group_count,
                              int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    for (int64_t i = 0; i < group_count; i++) {
        const auto new_side = left_right[i] == oneapi::mkl::side::left ? oneapi::mkl::side::right
                                                                       : oneapi::mkl::side::left;
        left_right[i] = new_side;

        const auto new_uplo = upper_lower[i] == oneapi::mkl::uplo::lower ? oneapi::mkl::uplo::upper
                                                                         : oneapi::mkl::uplo::lower;
        upper_lower[i] = new_uplo;
    }

    return column_major::trsm_batch(func, queue, left_right, upper_lower, trans, unit_diag, n, m,
                                    alpha, a, lda, b, ldb, group_count, group_size, dependencies);
}

#define TRSM_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                             \
    sycl::event trsm_batch(sycl::queue &queue, side *left_right, uplo *upper_lower,                \
                           transpose *trans, diag *unit_diag, int64_t *m, int64_t *n, TYPE *alpha, \
                           const TYPE **a, int64_t *lda, TYPE **b, int64_t *ldb,                   \
                           int64_t group_count, int64_t *group_size,                               \
                           const std::vector<sycl::event> &dependencies) {                         \
        return trsm_batch(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, \
                          alpha, a, lda, b, ldb, group_count, group_size, dependencies);           \
    }

TRSM_BATCH_LAUNCHER_USM(float, rocblas_strsm_batched)
TRSM_BATCH_LAUNCHER_USM(double, rocblas_dtrsm_batched)
TRSM_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_ctrsm_batched)
TRSM_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_ztrsm_batched)

#undef TRSM_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event syrk_batch(Func func, sycl::queue &queue, uplo *upper_lower, transpose *trans,
                              int64_t *n, int64_t *k, T *alpha, const T **a, int64_t *lda, T *beta,
                              T **c, int64_t *ldc, int64_t group_count, int64_t *group_size,
                              const std::vector<sycl::event> &dependencies) {
    for (int64_t i = 0; i < group_count; i++) {
        const auto new_uplo = upper_lower[i] == oneapi::mkl::uplo::lower ? oneapi::mkl::uplo::upper
                                                                         : oneapi::mkl::uplo::lower;
        upper_lower[i] = new_uplo;

        const auto new_trans = trans[i] == oneapi::mkl::transpose::nontrans
                                   ? oneapi::mkl::transpose::trans
                                   : oneapi::mkl::transpose::nontrans;
        trans[i] = new_trans;
    }

    return column_major::syrk_batch(func, queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                                    ldc, group_count, group_size, dependencies);
}

#define SYRK_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                           \
    sycl::event syrk_batch(sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,  \
                           int64_t *k, TYPE *alpha, const TYPE **a, int64_t *lda, TYPE *beta,    \
                           TYPE **c, int64_t *ldc, int64_t group_count, int64_t *group_size,     \
                           const std::vector<sycl::event> &dependencies) {                       \
        return syrk_batch(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, \
                          c, ldc, group_count, group_size, dependencies);                        \
    }

SYRK_BATCH_LAUNCHER_USM(float, rocblas_ssyrk_batched)
SYRK_BATCH_LAUNCHER_USM(double, rocblas_dsyrk_batched)
SYRK_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_csyrk_batched)
SYRK_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zsyrk_batched)

#undef SYRK_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event syrk_batch(Func func, sycl::queue &queue, uplo upper_lower, transpose trans,
                              int64_t n, int64_t k, const T alpha, const T *a, int64_t lda,
                              int64_t stridea, const T beta, T *c, int64_t ldc, int64_t stridec,
                              int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    auto new_uplo = upper_lower == oneapi::mkl::uplo::lower ? oneapi::mkl::uplo::upper
                                                            : oneapi::mkl::uplo::lower;
    auto new_trans = trans == oneapi::mkl::transpose::nontrans ? oneapi::mkl::transpose::trans
                                                               : oneapi::mkl::transpose::nontrans;

    return column_major::syrk_batch(func, queue, new_uplo, new_trans, n, k, alpha, a, lda, stridea,
                                    beta, c, ldc, stridec, batch_size, dependencies);
}

#define SYRK_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                               \
    sycl::event syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, \
                           int64_t k, const TYPE alpha, const TYPE *a, int64_t lda,          \
                           int64_t stridea, const TYPE beta, TYPE *c, int64_t ldc,           \
                           int64_t stridec, int64_t batch_size,                              \
                           const std::vector<sycl::event> &dependencies) {                   \
        return syrk_batch(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda,   \
                          stridea, beta, c, ldc, stridec, batch_size, dependencies);         \
    }

SYRK_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_ssyrk_strided_batched)
SYRK_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_dsyrk_strided_batched)
SYRK_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_csyrk_strided_batched)
SYRK_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zsyrk_strided_batched)

#undef SYRK_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event omatcopy_batch(Func func, sycl::queue &queue, transpose trans, int64_t m,
                                  int64_t n, const T alpha, const T *a, int64_t lda,
                                  int64_t stridea, T *b, int64_t ldb, int64_t strideb,
                                  int64_t batch_size,
                                  const std::vector<sycl::event> &dependencies) {
    return column_major::omatcopy_batch(func, queue, trans, n, m, alpha, a, lda, stridea, b, ldb,
                                        strideb, batch_size, dependencies);
}

#define OMATCOPY_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                 \
    sycl::event omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,          \
                               const TYPE alpha, const TYPE *a, int64_t lda, int64_t stridea,      \
                               TYPE *b, int64_t ldb, int64_t strideb, int64_t batch_size,          \
                               const std::vector<sycl::event> &dependencies) {                     \
        return omatcopy_batch(ROCBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, stridea, b, ldb, \
                              strideb, batch_size, dependencies);                                  \
    }

OMATCOPY_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_sgeam_strided_batched)
OMATCOPY_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_dgeam_strided_batched)
OMATCOPY_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cgeam_strided_batched)
OMATCOPY_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zgeam_strided_batched)

#undef OMATCOPY_STRIDED_BATCH_LAUNCHER_USM

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                           float *ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "row_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                           double *ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "row_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                           std::complex<float> alpha, std::complex<float> *ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "row_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                           std::complex<double> alpha, std::complex<double> *ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "row_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

template <typename Func, typename T>
inline sycl::event omatadd_batch(Func func, sycl::queue &queue, transpose transa, transpose transb,
                                 int64_t m, int64_t n, const T alpha, const T *a, int64_t lda,
                                 int64_t stridea, const T beta, const T *b, int64_t ldb,
                                 int64_t strideb, T *c, int64_t ldc, int64_t stridec,
                                 int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    return column_major::omatadd_batch(func, queue, transa, transb, n, m, alpha, a, lda, stridea,
                                       beta, b, ldb, strideb, c, ldc, stridec, batch_size,
                                       dependencies);
}

#define OMATADD_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                  \
    sycl::event omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,   \
                              int64_t n, const TYPE alpha, const TYPE *a, int64_t lda,             \
                              int64_t stridea, const TYPE beta, const TYPE *b, int64_t ldb,        \
                              int64_t strideb, TYPE *c, int64_t ldc, int64_t stridec,              \
                              int64_t batch_size, const std::vector<sycl::event> &dependencies) {  \
        return omatadd_batch(ROCBLAS_ROUTINE, queue, transa, transb, m, n, alpha, a, lda, stridea, \
                             beta, b, ldb, strideb, c, ldc, stridec, batch_size, dependencies);    \
    }

OMATADD_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_sgeam_strided_batched)
OMATADD_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_dgeam_strided_batched)
OMATADD_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cgeam_strided_batched)
OMATADD_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zgeam_strided_batched)

#undef OMATADD_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event omatcopy_batch(Func func, sycl::queue &queue, transpose *trans, int64_t *m,
                                  int64_t *n, T *alpha, const T **a, int64_t *lda, T **b,
                                  int64_t *ldb, int64_t group_count, int64_t *group_size,
                                  const std::vector<sycl::event> &dependencies) {
    return column_major::omatcopy_batch(func, queue, trans, n, m, alpha, a, lda, b, ldb,
                                        group_count, group_size, dependencies);
}

#define OMATCOPY_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                        \
    sycl::event omatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,      \
                               TYPE *alpha, const TYPE **a, int64_t *lda, TYPE **b, int64_t *ldb, \
                               int64_t group_count, int64_t *group_size,                          \
                               const std::vector<sycl::event> &dependencies) {                    \
        return omatcopy_batch(ROCBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, b, ldb,         \
                              group_count, group_size, dependencies);                             \
    }

OMATCOPY_BATCH_LAUNCHER_USM(float, rocblas_sgeam_batched)
OMATCOPY_BATCH_LAUNCHER_USM(double, rocblas_dgeam_batched)
OMATCOPY_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cgeam_batched)
OMATCOPY_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zgeam_batched)

#undef OMATCOPY_BATCH_LAUNCHER_USM

sycl::event imatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           float *alpha, float **ab, int64_t *lda, int64_t *ldb,
                           int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "row_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           double *alpha, double **ab, int64_t *lda, int64_t *ldb,
                           int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "row_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           std::complex<float> *alpha, std::complex<float> **ab, int64_t *lda,
                           int64_t *ldb, int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "row_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           std::complex<double> *alpha, std::complex<double> **ab, int64_t *lda,
                           int64_t *ldb, int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "row_major imatcopy_batch",
                        "imatcopy_batch unsupported by the rocBLAS API");
}

} // namespace row_major
} // namespace rocblas
} // namespace blas
} // namespace mkl
} // namespace oneapi
