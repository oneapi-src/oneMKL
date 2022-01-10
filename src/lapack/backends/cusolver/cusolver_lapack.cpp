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
#include "cusolver_helper.hpp"
#include "cusolver_task.hpp"

#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/lapack/detail/cusolver/onemkl_lapack_cusolver.hpp"

namespace oneapi {
namespace mkl {
namespace lapack {
namespace cusolver {

// BUFFER APIs

template <typename Func, typename T_A, typename T_B>
inline void gebrd(Func func, sycl::queue &queue, std::int64_t m, std::int64_t n,
                  sycl::buffer<T_A> &a, std::int64_t lda, sycl::buffer<T_B> &d,
                  sycl::buffer<T_B> &e, sycl::buffer<T_A> &tauq, sycl::buffer<T_A> &taup,
                  sycl::buffer<T_A> &scratchpad, std::int64_t scratchpad_size) {
    using cuDataType_A = typename CudaEquivalentType<T_A>::Type;
    using cuDataType_B = typename CudaEquivalentType<T_B>::Type;
    overflow_check(m, n, lda, scratchpad_size);

    if (m < n)
        throw unimplemented("lapack", "gebrd", "cusolver gebrd does not support m < n");

    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto d_acc = d.template get_access<cl::sycl::access::mode::write>(cgh);
        auto e_acc = e.template get_access<cl::sycl::access::mode::write>(cgh);
        auto tauq_acc = tauq.template get_access<cl::sycl::access::mode::write>(cgh);
        auto taup_acc = taup.template get_access<cl::sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType_A *>(a_acc);
            auto d_ = sc.get_mem<cuDataType_B *>(d_acc);
            auto e_ = sc.get_mem<cuDataType_B *>(e_acc);
            auto tauq_ = sc.get_mem<cuDataType_A *>(tauq_acc);
            auto taup_ = sc.get_mem<cuDataType_A *>(taup_acc);
            auto scratch_ = sc.get_mem<cuDataType_A *>(scratch_acc);
            cusolverStatus_t err;
            CUSOLVER_ERROR_FUNC(func, err, handle, m, n, a_, lda, d_, e_, tauq_, taup_, scratch_,
                                scratchpad_size, nullptr);
        });
    });
}

#define GEBRD_LAUNCHER(TYPE_A, TYPE_B, CUSOLVER_ROUTINE)                                    \
    void gebrd(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<TYPE_A> &a, \
               std::int64_t lda, sycl::buffer<TYPE_B> &d, sycl::buffer<TYPE_B> &e,          \
               sycl::buffer<TYPE_A> &tauq, sycl::buffer<TYPE_A> &taup,                      \
               sycl::buffer<TYPE_A> &scratchpad, std::int64_t scratchpad_size) {            \
        gebrd(CUSOLVER_ROUTINE, queue, m, n, a, lda, d, e, tauq, taup, scratchpad,          \
              scratchpad_size);                                                             \
    }

GEBRD_LAUNCHER(float, float, cusolverDnSgebrd)
GEBRD_LAUNCHER(double, double, cusolverDnDgebrd)
GEBRD_LAUNCHER(std::complex<float>, float, cusolverDnCgebrd)
GEBRD_LAUNCHER(std::complex<double>, double, cusolverDnZgebrd)

#undef GEBRD_LAUNCHER

void gerqf(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<float> &a,
           std::int64_t lda, sycl::buffer<float> &tau, sycl::buffer<float> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gerqf");
}
void gerqf(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<double> &a,
           std::int64_t lda, sycl::buffer<double> &tau, sycl::buffer<double> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gerqf");
}
void gerqf(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<std::complex<float>> &a,
           std::int64_t lda, sycl::buffer<std::complex<float>> &tau,
           sycl::buffer<std::complex<float>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gerqf");
}
void gerqf(sycl::queue &queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>> &tau, sycl::buffer<std::complex<double>> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gerqf");
}

template <typename Func, typename T>
inline void geqrf(Func func, sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<T> &a,
                  std::int64_t lda, sycl::buffer<T> &tau, sycl::buffer<T> &scratchpad,
                  std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, scratchpad_size);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<cl::sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType *>(a_acc);
            auto tau_ = sc.get_mem<cuDataType *>(tau_acc);
            auto scratch_ = sc.get_mem<cuDataType *>(scratch_acc);
            cusolverStatus_t err;
            CUSOLVER_ERROR_FUNC(func, err, handle, m, n, a_, lda, tau_, scratch_, scratchpad_size,
                                nullptr);
        });
    });
}

#define GEQRF_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                            \
    void geqrf(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<TYPE> &a, \
               std::int64_t lda, sycl::buffer<TYPE> &tau, sycl::buffer<TYPE> &scratchpad, \
               std::int64_t scratchpad_size) {                                            \
        geqrf(CUSOLVER_ROUTINE, queue, m, n, a, lda, tau, scratchpad, scratchpad_size);   \
    }

GEQRF_LAUNCHER(float, cusolverDnSgeqrf)
GEQRF_LAUNCHER(double, cusolverDnDgeqrf)
GEQRF_LAUNCHER(std::complex<float>, cusolverDnCgeqrf)
GEQRF_LAUNCHER(std::complex<double>, cusolverDnZgeqrf)

#undef GEQRF_LAUNCHER

template <typename Func, typename T>
void getrf(Func func, sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<T> &a,
           std::int64_t lda, sycl::buffer<std::int64_t> &ipiv, sycl::buffer<T> &scratchpad,
           std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, scratchpad_size);

    // cusolver legacy api does not accept 64-bit ints
    // Create new buffer with 32-bit ints then copy over
    std::uint64_t ipiv_size = std::min(n, m);
    sycl::buffer<int, 1> ipiv32(sycl::range<1>{ ipiv_size });

    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto ipiv32_acc = ipiv32.template get_access<cl::sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<cl::sycl::access::mode::write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType *>(a_acc);
            auto ipiv32_ = sc.get_mem<int *>(ipiv32_acc);
            auto scratch_ = sc.get_mem<cuDataType *>(scratch_acc);
            cusolverStatus_t err;
            CUSOLVER_ERROR_FUNC(func, err, handle, m, n, a_, lda, scratch_, ipiv32_, nullptr);
        });
    });

    // Copy from 32-bit buffer to 64-bit
    queue.submit([&](cl::sycl::handler &cgh) {
        auto ipiv32_acc = ipiv32.template get_access<cl::sycl::access::mode::read>(cgh);
        auto ipiv_acc = ipiv.template get_access<cl::sycl::access::mode::write>(cgh);
        cgh.parallel_for(cl::sycl::range<1>{ ipiv_size }, [=](cl::sycl::id<1> index) {
            ipiv_acc[index] = static_cast<std::int64_t>(ipiv32_acc[index]);
        });
    });
}

#define GETRF_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                                     \
    void getrf(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<TYPE> &a,          \
               std::int64_t lda, sycl::buffer<std::int64_t> &ipiv, sycl::buffer<TYPE> &scratchpad, \
               std::int64_t scratchpad_size) {                                                     \
        getrf(CUSOLVER_ROUTINE, queue, m, n, a, lda, ipiv, scratchpad, scratchpad_size);           \
    }

GETRF_LAUNCHER(float, cusolverDnSgetrf)
GETRF_LAUNCHER(double, cusolverDnDgetrf)
GETRF_LAUNCHER(std::complex<float>, cusolverDnCgetrf)
GETRF_LAUNCHER(std::complex<double>, cusolverDnZgetrf)

#undef GETRF_LAUNCHER

void getri(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>> &a,
           std::int64_t lda, sycl::buffer<std::int64_t> &ipiv,
           sycl::buffer<std::complex<float>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri");
}
void getri(sycl::queue &queue, std::int64_t n, sycl::buffer<double> &a, std::int64_t lda,
           sycl::buffer<std::int64_t> &ipiv, sycl::buffer<double> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri");
}
void getri(sycl::queue &queue, std::int64_t n, sycl::buffer<float> &a, std::int64_t lda,
           sycl::buffer<std::int64_t> &ipiv, sycl::buffer<float> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri");
}
void getri(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>> &a,
           std::int64_t lda, sycl::buffer<std::int64_t> &ipiv,
           sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri");
}

// cusolverDnXgetrs does not use scratchpad memory
template <typename Func, typename T>
inline void getrs(Func func, sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n,
                  std::int64_t nrhs, sycl::buffer<T> &a, std::int64_t lda,
                  sycl::buffer<std::int64_t> &ipiv, sycl::buffer<T> &b, std::int64_t ldb,
                  sycl::buffer<T> &scratchpad, std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, nrhs, lda, ldb);

    //cusolver does not support int64, so ipiv must be converted to int32 before
    //being passed to cusolverDnXgetrs.
    std::uint64_t ipiv_size = ipiv.size();
    sycl::buffer<int, 1> ipiv32(sycl::range<1>{ ipiv_size });

    queue.submit([&](cl::sycl::handler &cgh) {
        auto ipiv32_acc = ipiv32.template get_access<cl::sycl::access::mode::write>(cgh);
        auto ipiv_acc = ipiv.template get_access<cl::sycl::access::mode::read>(cgh);
        cgh.parallel_for(cl::sycl::range<1>{ ipiv_size },
                         [=](cl::sycl::id<1> index) { ipiv32_acc[index] = ipiv_acc[index]; });
    });

    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto ipiv_acc = ipiv32.template get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType *>(a_acc);
            auto ipiv_ = sc.get_mem<std::int32_t *>(ipiv_acc);
            auto b_ = sc.get_mem<cuDataType *>(b_acc);
            cusolverStatus_t err;
            CUSOLVER_ERROR_FUNC(func, err, handle, get_cublas_operation(trans), n, nrhs, a_, lda,
                                ipiv_, b_, ldb, nullptr);
        });
    });
}

#define GETRS_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                            \
    void getrs(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n,          \
               std::int64_t nrhs, sycl::buffer<TYPE> &a, std::int64_t lda,                \
               sycl::buffer<std::int64_t> &ipiv, sycl::buffer<TYPE> &b, std::int64_t ldb, \
               sycl::buffer<TYPE> &scratchpad, std::int64_t scratchpad_size) {            \
        getrs(CUSOLVER_ROUTINE, queue, trans, n, nrhs, a, lda, ipiv, b, ldb, scratchpad,  \
              scratchpad_size);                                                           \
    }

GETRS_LAUNCHER(float, cusolverDnSgetrs)
GETRS_LAUNCHER(double, cusolverDnDgetrs)
GETRS_LAUNCHER(std::complex<float>, cusolverDnCgetrs)
GETRS_LAUNCHER(std::complex<double>, cusolverDnZgetrs)

#undef GETRS_LAUNCHER

template <typename Func, typename T_A, typename T_B>
inline void gesvd(Func func, sycl::queue &queue, oneapi::mkl::jobsvd jobu,
                  oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n, sycl::buffer<T_A> &a,
                  std::int64_t lda, sycl::buffer<T_B> &s, sycl::buffer<T_A> &u, std::int64_t ldu,
                  sycl::buffer<T_A> &vt, std::int64_t ldvt, sycl::buffer<T_A> &scratchpad,
                  std::int64_t scratchpad_size) {
    using cuDataType_A = typename CudaEquivalentType<T_A>::Type;
    using cuDataType_B = typename CudaEquivalentType<T_B>::Type;
    overflow_check(n, m, lda, ldu, ldvt, scratchpad_size);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto s_acc = s.template get_access<cl::sycl::access::mode::write>(cgh);
        auto u_acc = u.template get_access<cl::sycl::access::mode::write>(cgh);
        auto vt_acc = vt.template get_access<cl::sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType_A *>(a_acc);
            auto s_ = sc.get_mem<cuDataType_B *>(s_acc);
            auto u_ = sc.get_mem<cuDataType_A *>(u_acc);
            auto vt_ = sc.get_mem<cuDataType_A *>(vt_acc);
            auto scratch_ = sc.get_mem<cuDataType_A *>(scratch_acc);
            cusolverStatus_t err;
            // rwork is set to nullptr. If set it is filled with information from the superdiagonal.
            CUSOLVER_ERROR_FUNC(func, err, handle, get_cusolver_jobsvd(jobu),
                                get_cusolver_jobsvd(jobvt), m, n, a_, lda, s_, u_, ldu, vt_, ldvt,
                                scratch_, scratchpad_size, nullptr, nullptr);
        });
    });
}

#define GESVD_LAUNCHER(TYPE_A, TYPE_B, CUSOLVER_ROUTINE)                                           \
    void gesvd(sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt,            \
               std::int64_t m, std::int64_t n, sycl::buffer<TYPE_A> &a, std::int64_t lda,          \
               sycl::buffer<TYPE_B> &s, sycl::buffer<TYPE_A> &u, std::int64_t ldu,                 \
               sycl::buffer<TYPE_A> &vt, std::int64_t ldvt, sycl::buffer<TYPE_A> &scratchpad,      \
               std::int64_t scratchpad_size) {                                                     \
        gesvd(CUSOLVER_ROUTINE, queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, scratchpad, \
              scratchpad_size);                                                                    \
    }

GESVD_LAUNCHER(float, float, cusolverDnSgesvd)
GESVD_LAUNCHER(double, double, cusolverDnDgesvd)
GESVD_LAUNCHER(std::complex<float>, float, cusolverDnCgesvd)
GESVD_LAUNCHER(std::complex<double>, double, cusolverDnZgesvd)

#undef GESVD_LAUNCHER

void heevd(sycl::queue &queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>> &a, std::int64_t lda, sycl::buffer<float> &w,
           sycl::buffer<std::complex<float>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "heevd");
}
void heevd(sycl::queue &queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>> &a, std::int64_t lda, sycl::buffer<double> &w,
           sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "heevd");
}
void hegvd(sycl::queue &queue, std::int64_t itype, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
           std::int64_t n, sycl::buffer<std::complex<float>> &a, std::int64_t lda,
           sycl::buffer<std::complex<float>> &b, std::int64_t ldb, sycl::buffer<float> &w,
           sycl::buffer<std::complex<float>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "hegvd");
}
void hegvd(sycl::queue &queue, std::int64_t itype, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
           std::int64_t n, sycl::buffer<std::complex<double>> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>> &b, std::int64_t ldb, sycl::buffer<double> &w,
           sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "hegvd");
}
void hetrd(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>> &a, std::int64_t lda, sycl::buffer<float> &d,
           sycl::buffer<float> &e, sycl::buffer<std::complex<float>> &tau,
           sycl::buffer<std::complex<float>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "hetrd");
}
void hetrd(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>> &a, std::int64_t lda, sycl::buffer<double> &d,
           sycl::buffer<double> &e, sycl::buffer<std::complex<double>> &tau,
           sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "hetrd");
}
void hetrf(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>> &a, std::int64_t lda, sycl::buffer<std::int64_t> &ipiv,
           sycl::buffer<std::complex<float>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "hetrf");
}
void hetrf(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>> &a, std::int64_t lda,
           sycl::buffer<std::int64_t> &ipiv, sycl::buffer<std::complex<double>> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "hetrf");
}
void orgbr(sycl::queue &queue, oneapi::mkl::generate vec, std::int64_t m, std::int64_t n,
           std::int64_t k, sycl::buffer<float> &a, std::int64_t lda, sycl::buffer<float> &tau,
           sycl::buffer<float> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "orgbr");
}
void orgbr(sycl::queue &queue, oneapi::mkl::generate vec, std::int64_t m, std::int64_t n,
           std::int64_t k, sycl::buffer<double> &a, std::int64_t lda, sycl::buffer<double> &tau,
           sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "orgbr");
}
void orgqr(sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<double> &a, std::int64_t lda, sycl::buffer<double> &tau,
           sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "orgqr");
}
void orgqr(sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<float> &a, std::int64_t lda, sycl::buffer<float> &tau,
           sycl::buffer<float> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "orgqr");
}
void orgtr(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<float> &a,
           std::int64_t lda, sycl::buffer<float> &tau, sycl::buffer<float> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "orgtr");
}
void orgtr(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<double> &a,
           std::int64_t lda, sycl::buffer<double> &tau, sycl::buffer<double> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "orgtr");
}
void ormtr(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo,
           oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, sycl::buffer<float> &a,
           std::int64_t lda, sycl::buffer<float> &tau, sycl::buffer<float> &c, std::int64_t ldc,
           sycl::buffer<float> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ormtr");
}
void ormtr(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo,
           oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, sycl::buffer<double> &a,
           std::int64_t lda, sycl::buffer<double> &tau, sycl::buffer<double> &c, std::int64_t ldc,
           sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ormtr");
}
void ormrq(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m,
           std::int64_t n, std::int64_t k, sycl::buffer<float> &a, std::int64_t lda,
           sycl::buffer<float> &tau, sycl::buffer<float> &c, std::int64_t ldc,
           sycl::buffer<float> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ormrq");
}
void ormrq(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m,
           std::int64_t n, std::int64_t k, sycl::buffer<double> &a, std::int64_t lda,
           sycl::buffer<double> &tau, sycl::buffer<double> &c, std::int64_t ldc,
           sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ormrq");
}
void ormqr(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m,
           std::int64_t n, std::int64_t k, sycl::buffer<double> &a, std::int64_t lda,
           sycl::buffer<double> &tau, sycl::buffer<double> &c, std::int64_t ldc,
           sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ormqr");
}
void ormqr(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m,
           std::int64_t n, std::int64_t k, sycl::buffer<float> &a, std::int64_t lda,
           sycl::buffer<float> &tau, sycl::buffer<float> &c, std::int64_t ldc,
           sycl::buffer<float> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ormqr");
}

template <typename Func, typename T>
inline void potrf(Func func, sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                  sycl::buffer<T> &a, std::int64_t lda, sycl::buffer<T> &scratchpad,
                  std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto scratch_acc = scratchpad.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType *>(a_acc);
            auto scratch_ = sc.get_mem<cuDataType *>(scratch_acc);
            cusolverStatus_t err;
            CUSOLVER_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(uplo), n, a_, lda, scratch_,
                                scratchpad_size, nullptr);
        });
    });
}

#define POTRF_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                                    \
    void potrf(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<TYPE> &a, \
               std::int64_t lda, sycl::buffer<TYPE> &scratchpad, std::int64_t scratchpad_size) {  \
        potrf(CUSOLVER_ROUTINE, queue, uplo, n, a, lda, scratchpad, scratchpad_size);             \
    }

POTRF_LAUNCHER(float, cusolverDnSpotrf)
POTRF_LAUNCHER(double, cusolverDnDpotrf)
POTRF_LAUNCHER(std::complex<float>, cusolverDnCpotrf)
POTRF_LAUNCHER(std::complex<double>, cusolverDnZpotrf)

#undef POTRF_LAUNCHER

void potri(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<float> &a,
           std::int64_t lda, sycl::buffer<float> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potri");
}
void potri(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<double> &a,
           std::int64_t lda, sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potri");
}
void potri(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>> &a, std::int64_t lda,
           sycl::buffer<std::complex<float>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potri");
}
void potri(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potri");
}
void potrs(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<float> &a, std::int64_t lda, sycl::buffer<float> &b, std::int64_t ldb,
           sycl::buffer<float> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrs");
}
void potrs(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<double> &a, std::int64_t lda, sycl::buffer<double> &b, std::int64_t ldb,
           sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrs");
}
void potrs(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<std::complex<float>> &a, std::int64_t lda,
           sycl::buffer<std::complex<float>> &b, std::int64_t ldb,
           sycl::buffer<std::complex<float>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrs");
}
void potrs(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<std::complex<double>> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>> &b, std::int64_t ldb,
           sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrs");
}
void syevd(sycl::queue &queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<double> &a, std::int64_t lda, sycl::buffer<double> &w,
           sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "syevd");
}
void syevd(sycl::queue &queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<float> &a, std::int64_t lda, sycl::buffer<float> &w,
           sycl::buffer<float> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "syevd");
}
void sygvd(sycl::queue &queue, std::int64_t itype, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
           std::int64_t n, sycl::buffer<double> &a, std::int64_t lda, sycl::buffer<double> &b,
           std::int64_t ldb, sycl::buffer<double> &w, sycl::buffer<double> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "sygvd");
}
void sygvd(sycl::queue &queue, std::int64_t itype, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
           std::int64_t n, sycl::buffer<float> &a, std::int64_t lda, sycl::buffer<float> &b,
           std::int64_t ldb, sycl::buffer<float> &w, sycl::buffer<float> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "sygvd");
}
void sytrd(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<double> &a,
           std::int64_t lda, sycl::buffer<double> &d, sycl::buffer<double> &e,
           sycl::buffer<double> &tau, sycl::buffer<double> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "sytrd");
}
void sytrd(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<float> &a,
           std::int64_t lda, sycl::buffer<float> &d, sycl::buffer<float> &e,
           sycl::buffer<float> &tau, sycl::buffer<float> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "sytrd");
}
void sytrf(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<float> &a,
           std::int64_t lda, sycl::buffer<std::int64_t> &ipiv, sycl::buffer<float> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "sytrf");
}
void sytrf(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<double> &a,
           std::int64_t lda, sycl::buffer<std::int64_t> &ipiv, sycl::buffer<double> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "sytrf");
}
void sytrf(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>> &a, std::int64_t lda, sycl::buffer<std::int64_t> &ipiv,
           sycl::buffer<std::complex<float>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "sytrf");
}
void sytrf(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>> &a, std::int64_t lda,
           sycl::buffer<std::int64_t> &ipiv, sycl::buffer<std::complex<double>> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "sytrf");
}
void trtrs(sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
           oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<std::complex<float>> &a, std::int64_t lda,
           sycl::buffer<std::complex<float>> &b, std::int64_t ldb,
           sycl::buffer<std::complex<float>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "trtrs");
}
void trtrs(sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
           oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs, sycl::buffer<double> &a,
           std::int64_t lda, sycl::buffer<double> &b, std::int64_t ldb,
           sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "trtrs");
}
void trtrs(sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
           oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs, sycl::buffer<float> &a,
           std::int64_t lda, sycl::buffer<float> &b, std::int64_t ldb,
           sycl::buffer<float> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "trtrs");
}
void trtrs(sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
           oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<std::complex<double>> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>> &b, std::int64_t ldb,
           sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "trtrs");
}
void ungbr(sycl::queue &queue, oneapi::mkl::generate vec, std::int64_t m, std::int64_t n,
           std::int64_t k, sycl::buffer<std::complex<float>> &a, std::int64_t lda,
           sycl::buffer<std::complex<float>> &tau, sycl::buffer<std::complex<float>> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ungbr");
}
void ungbr(sycl::queue &queue, oneapi::mkl::generate vec, std::int64_t m, std::int64_t n,
           std::int64_t k, sycl::buffer<std::complex<double>> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>> &tau, sycl::buffer<std::complex<double>> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ungbr");
}
void ungqr(sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<std::complex<float>> &a, std::int64_t lda,
           sycl::buffer<std::complex<float>> &tau, sycl::buffer<std::complex<float>> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ungqr");
}
void ungqr(sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<std::complex<double>> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>> &tau, sycl::buffer<std::complex<double>> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ungqr");
}
void ungtr(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>> &a, std::int64_t lda,
           sycl::buffer<std::complex<float>> &tau, sycl::buffer<std::complex<float>> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ungtr");
}
void ungtr(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>> &tau, sycl::buffer<std::complex<double>> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ungtr");
}
void unmrq(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m,
           std::int64_t n, std::int64_t k, sycl::buffer<std::complex<float>> &a, std::int64_t lda,
           sycl::buffer<std::complex<float>> &tau, sycl::buffer<std::complex<float>> &c,
           std::int64_t ldc, sycl::buffer<std::complex<float>> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "unmrq");
}
void unmrq(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m,
           std::int64_t n, std::int64_t k, sycl::buffer<std::complex<double>> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>> &tau, sycl::buffer<std::complex<double>> &c,
           std::int64_t ldc, sycl::buffer<std::complex<double>> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "unmrq");
}
void unmqr(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m,
           std::int64_t n, std::int64_t k, sycl::buffer<std::complex<float>> &a, std::int64_t lda,
           sycl::buffer<std::complex<float>> &tau, sycl::buffer<std::complex<float>> &c,
           std::int64_t ldc, sycl::buffer<std::complex<float>> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "unmqr");
}
void unmqr(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m,
           std::int64_t n, std::int64_t k, sycl::buffer<std::complex<double>> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>> &tau, sycl::buffer<std::complex<double>> &c,
           std::int64_t ldc, sycl::buffer<std::complex<double>> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "unmqr");
}
void unmtr(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo,
           oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<float>> &a, std::int64_t lda,
           sycl::buffer<std::complex<float>> &tau, sycl::buffer<std::complex<float>> &c,
           std::int64_t ldc, sycl::buffer<std::complex<float>> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "unmtr");
}
void unmtr(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo,
           oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>> &tau, sycl::buffer<std::complex<double>> &c,
           std::int64_t ldc, sycl::buffer<std::complex<double>> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "unmtr");
}

// USM APIs

template <typename Func, typename T_A, typename T_B>
inline sycl::event gebrd(Func func, sycl::queue &queue, std::int64_t m, std::int64_t n, T_A *a,
                         std::int64_t lda, T_B *d, T_B *e, T_A *tauq, T_A *taup, T_A *scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event> &dependencies) {
    using cuDataType_A = typename CudaEquivalentType<T_A>::Type;
    using cuDataType_B = typename CudaEquivalentType<T_B>::Type;
    overflow_check(m, n, lda, scratchpad_size);

    if (m < n)
        throw unimplemented("lapack", "gebrd", "cusolver gebrd does not support m < n");

    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType_A *>(a);
            auto d_ = reinterpret_cast<cuDataType_B *>(d);
            auto e_ = reinterpret_cast<cuDataType_B *>(e);
            auto tauq_ = reinterpret_cast<cuDataType_A *>(tauq);
            auto taup_ = reinterpret_cast<cuDataType_A *>(taup);
            auto scratch_ = reinterpret_cast<cuDataType_A *>(scratchpad);
            cusolverStatus_t err;
            CUSOLVER_ERROR_FUNC(func, err, handle, m, n, a_, lda, d_, e_, tauq_, taup_, scratch_,
                                scratchpad_size, nullptr);
        });
    });
    return done;
}

#define GEBRD_LAUNCHER_USM(TYPE_A, TYPE_B, CUSOLVER_ROUTINE)                              \
    sycl::event gebrd(sycl::queue &queue, std::int64_t m, std::int64_t n, TYPE_A *a,      \
                      std::int64_t lda, TYPE_B *d, TYPE_B *e, TYPE_A *tauq, TYPE_A *taup, \
                      TYPE_A *scratchpad, std::int64_t scratchpad_size,                   \
                      const std::vector<sycl::event> &dependencies) {                     \
        return gebrd(CUSOLVER_ROUTINE, queue, m, n, a, lda, d, e, tauq, taup, scratchpad, \
                     scratchpad_size, dependencies);                                      \
    }

GEBRD_LAUNCHER_USM(float, float, cusolverDnSgebrd)
GEBRD_LAUNCHER_USM(double, double, cusolverDnDgebrd)
GEBRD_LAUNCHER_USM(std::complex<float>, float, cusolverDnCgebrd)
GEBRD_LAUNCHER_USM(std::complex<double>, double, cusolverDnZgebrd)

#undef GEBRD_LAUNCHER_USM

sycl::event gerqf(sycl::queue &queue, std::int64_t m, std::int64_t n, float *a, std::int64_t lda,
                  float *tau, float *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "gerqf");
}
sycl::event gerqf(sycl::queue &queue, std::int64_t m, std::int64_t n, double *a, std::int64_t lda,
                  double *tau, double *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "gerqf");
}
sycl::event gerqf(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> *a,
                  std::int64_t lda, std::complex<float> *tau, std::complex<float> *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "gerqf");
}
sycl::event gerqf(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> *a,
                  std::int64_t lda, std::complex<double> *tau, std::complex<double> *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "gerqf");
}

template <typename Func, typename T>
inline sycl::event geqrf(Func func, sycl::queue &queue, std::int64_t m, std::int64_t n, T *a,
                         std::int64_t lda, T *tau, T *scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, scratchpad_size);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType *>(a);
            auto tau_ = reinterpret_cast<cuDataType *>(tau);
            auto scratch_ = reinterpret_cast<cuDataType *>(scratchpad);
            cusolverStatus_t err;
            CUSOLVER_ERROR_FUNC(func, err, handle, m, n, a_, lda, tau_, scratch_, scratchpad_size,
                                nullptr);
        });
    });
    return done;
}

#define GEQRF_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                                 \
    sycl::event geqrf(sycl::queue &queue, std::int64_t m, std::int64_t n, TYPE *a,                 \
                      std::int64_t lda, TYPE *tau, TYPE *scratchpad, std::int64_t scratchpad_size, \
                      const std::vector<sycl::event> &dependencies) {                              \
        return geqrf(CUSOLVER_ROUTINE, queue, m, n, a, lda, tau, scratchpad, scratchpad_size,      \
                     dependencies);                                                                \
    }

GEQRF_LAUNCHER_USM(float, cusolverDnSgeqrf)
GEQRF_LAUNCHER_USM(double, cusolverDnDgeqrf)
GEQRF_LAUNCHER_USM(std::complex<float>, cusolverDnCgeqrf)
GEQRF_LAUNCHER_USM(std::complex<double>, cusolverDnZgeqrf)

#undef GEQRF_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event getrf(Func func, sycl::queue &queue, std::int64_t m, std::int64_t n, T *a,
                         std::int64_t lda, std::int64_t *ipiv, T *scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, scratchpad_size);

    // cusolver legacy api does not accept 64-bit ints
    // Create USM with 32-bit ints then copy over
    std::uint64_t ipiv_size = std::min(n, m);
    int *ipiv32 = (int *)malloc_device(ipiv_size, queue);

    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType *>(a);
            auto scratch_ = reinterpret_cast<cuDataType *>(scratchpad);
            auto ipiv_ = reinterpret_cast<int *>(ipiv32);
            cusolverStatus_t err;
            CUSOLVER_ERROR_FUNC(func, err, handle, m, n, a_, lda, scratch_, ipiv_, nullptr);
        });
    });

    // Copy from 32-bit USM to 64-bit
    queue.submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(cl::sycl::range<1>{ ipiv_size },
                         [=](cl::sycl::id<1> index) { ipiv[index] = ipiv32[index]; });
    });

    free(ipiv32, queue);

    return done;
}

#define GETRF_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                             \
    sycl::event getrf(sycl::queue &queue, std::int64_t m, std::int64_t n, TYPE *a,             \
                      std::int64_t lda, std::int64_t *ipiv, TYPE *scratchpad,                  \
                      std::int64_t scratchpad_size,                                            \
                      const std::vector<sycl::event> &dependencies) {                          \
        return getrf(CUSOLVER_ROUTINE, queue, m, n, a, lda, ipiv, scratchpad, scratchpad_size, \
                     dependencies);                                                            \
    }

GETRF_LAUNCHER_USM(float, cusolverDnSgetrf)
GETRF_LAUNCHER_USM(double, cusolverDnDgetrf)
GETRF_LAUNCHER_USM(std::complex<float>, cusolverDnCgetrf)
GETRF_LAUNCHER_USM(std::complex<double>, cusolverDnZgetrf)

#undef GETRF_LAUNCHER_USM

sycl::event getri(sycl::queue &queue, std::int64_t n, std::complex<float> *a, std::int64_t lda,
                  std::int64_t *ipiv, std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getri");
}
sycl::event getri(sycl::queue &queue, std::int64_t n, double *a, std::int64_t lda,
                  std::int64_t *ipiv, double *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getri");
}
sycl::event getri(sycl::queue &queue, std::int64_t n, float *a, std::int64_t lda,
                  std::int64_t *ipiv, float *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getri");
}
sycl::event getri(sycl::queue &queue, std::int64_t n, std::complex<double> *a, std::int64_t lda,
                  std::int64_t *ipiv, std::complex<double> *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getri");
}

// cusolverDnXgetrs does not use scratchpad memory
template <typename Func, typename T>
inline sycl::event getrs(Func func, sycl::queue &queue, oneapi::mkl::transpose trans,
                         std::int64_t n, std::int64_t nrhs, T *a, std::int64_t lda,
                         std::int64_t *ipiv, T *b, std::int64_t ldb, T *scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, nrhs, lda, ldb, scratchpad_size);

    //cusolver does not support int64, so ipiv must be converted to int32 before
    //being passed to cusolverDnXgetrs.
    std::uint64_t ipiv_size = n;
    int *ipiv32 = (int *)malloc_device(ipiv_size, queue);

    queue.submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(cl::sycl::range<1>{ ipiv_size },
                         [=](cl::sycl::id<1> index) { ipiv32[index] = ipiv[index]; });
    });

    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType *>(a);
            auto ipiv_ = reinterpret_cast<int *>(ipiv32);
            auto b_ = reinterpret_cast<cuDataType *>(b);
            cusolverStatus_t err;
            CUSOLVER_ERROR_FUNC(func, err, handle, get_cublas_operation(trans), n, nrhs, a_, lda,
                                ipiv_, b_, ldb, nullptr);
        });
    });

    free(ipiv32, queue);

    return done;
}

#define GETRS_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                               \
    sycl::event getrs(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n,          \
                      std::int64_t nrhs, TYPE *a, std::int64_t lda, std::int64_t *ipiv, TYPE *b, \
                      std::int64_t ldb, TYPE *scratchpad, std::int64_t scratchpad_size,          \
                      const std::vector<sycl::event> &dependencies) {                            \
        return getrs(CUSOLVER_ROUTINE, queue, trans, n, nrhs, a, lda, ipiv, b, ldb, scratchpad,  \
                     scratchpad_size, dependencies);                                             \
    }

GETRS_LAUNCHER_USM(float, cusolverDnSgetrs)
GETRS_LAUNCHER_USM(double, cusolverDnDgetrs)
GETRS_LAUNCHER_USM(std::complex<float>, cusolverDnCgetrs)
GETRS_LAUNCHER_USM(std::complex<double>, cusolverDnZgetrs)

#undef GETRS_LAUNCHER_USM

template <typename Func, typename T_A, typename T_B>
inline sycl::event gesvd(Func func, sycl::queue &queue, oneapi::mkl::jobsvd jobu,
                         oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n, T_A *a,
                         std::int64_t lda, T_B *s, T_A *u, std::int64_t ldu, T_A *vt,
                         std::int64_t ldvt, T_A *scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event> &dependencies) {
    using cuDataType_A = typename CudaEquivalentType<T_A>::Type;
    using cuDataType_B = typename CudaEquivalentType<T_B>::Type;
    overflow_check(m, n, lda, ldu, ldvt, scratchpad_size);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType_A *>(a);
            auto s_ = reinterpret_cast<cuDataType_B *>(s);
            auto u_ = reinterpret_cast<cuDataType_A *>(u);
            auto vt_ = reinterpret_cast<cuDataType_A *>(vt);
            auto scratch_ = reinterpret_cast<cuDataType_A *>(scratchpad);
            cusolverStatus_t err;
            // rwork is set to nullptr. If set it is filled with information from the superdiagonal.
            CUSOLVER_ERROR_FUNC(func, err, handle, get_cusolver_jobsvd(jobu),
                                get_cusolver_jobsvd(jobvt), m, n, a_, lda, s_, u_, ldu, vt_, ldvt,
                                scratch_, scratchpad_size, nullptr, nullptr);
        });
    });
    return done;
}

#define GESVD_LAUNCHER_USM(TYPE_A, TYPE_B, CUSOLVER_ROUTINE)                                   \
    sycl::event gesvd(sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, \
                      std::int64_t m, std::int64_t n, TYPE_A *a, std::int64_t lda, TYPE_B *s,  \
                      TYPE_A *u, std::int64_t ldu, TYPE_A *vt, std::int64_t ldvt,              \
                      TYPE_A *scratchpad, std::int64_t scratchpad_size,                        \
                      const std::vector<sycl::event> &dependencies) {                          \
        return gesvd(CUSOLVER_ROUTINE, queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,  \
                     scratchpad, scratchpad_size, dependencies);                               \
    }

GESVD_LAUNCHER_USM(float, float, cusolverDnSgesvd)
GESVD_LAUNCHER_USM(double, double, cusolverDnDgesvd)
GESVD_LAUNCHER_USM(std::complex<float>, float, cusolverDnCgesvd)
GESVD_LAUNCHER_USM(std::complex<double>, double, cusolverDnZgesvd)

#undef GESVD_LAUNCHER_USM

sycl::event heevd(sycl::queue &queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                  std::complex<float> *a, std::int64_t lda, float *w,
                  std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "heevd");
}
sycl::event heevd(sycl::queue &queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                  std::complex<double> *a, std::int64_t lda, double *w,
                  std::complex<double> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "heevd");
}
sycl::event hegvd(sycl::queue &queue, std::int64_t itype, oneapi::mkl::job jobz,
                  oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float> *a, std::int64_t lda,
                  std::complex<float> *b, std::int64_t ldb, float *w,
                  std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "hegvd");
}
sycl::event hegvd(sycl::queue &queue, std::int64_t itype, oneapi::mkl::job jobz,
                  oneapi::mkl::uplo uplo, std::int64_t n, std::complex<double> *a, std::int64_t lda,
                  std::complex<double> *b, std::int64_t ldb, double *w,
                  std::complex<double> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "hegvd");
}
sycl::event hetrd(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                  std::complex<float> *a, std::int64_t lda, float *d, float *e,
                  std::complex<float> *tau, std::complex<float> *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "hetrd");
}
sycl::event hetrd(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                  std::complex<double> *a, std::int64_t lda, double *d, double *e,
                  std::complex<double> *tau, std::complex<double> *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "hetrd");
}
sycl::event hetrf(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                  std::complex<float> *a, std::int64_t lda, std::int64_t *ipiv,
                  std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "hetrf");
}
sycl::event hetrf(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                  std::complex<double> *a, std::int64_t lda, std::int64_t *ipiv,
                  std::complex<double> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "hetrf");
}
sycl::event orgbr(sycl::queue &queue, oneapi::mkl::generate vec, std::int64_t m, std::int64_t n,
                  std::int64_t k, float *a, std::int64_t lda, float *tau, float *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "orgbr");
}
sycl::event orgbr(sycl::queue &queue, oneapi::mkl::generate vec, std::int64_t m, std::int64_t n,
                  std::int64_t k, double *a, std::int64_t lda, double *tau, double *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "orgbr");
}
sycl::event orgqr(sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, double *a,
                  std::int64_t lda, double *tau, double *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "orgqr");
}
sycl::event orgqr(sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, float *a,
                  std::int64_t lda, float *tau, float *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "orgqr");
}
sycl::event orgtr(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float *a,
                  std::int64_t lda, float *tau, float *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "orgtr");
}
sycl::event orgtr(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double *a,
                  std::int64_t lda, double *tau, double *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "orgtr");
}
sycl::event ormtr(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                  oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, float *a,
                  std::int64_t lda, float *tau, float *c, std::int64_t ldc, float *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "ormtr");
}
sycl::event ormtr(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                  oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, double *a,
                  std::int64_t lda, double *tau, double *c, std::int64_t ldc, double *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "ormtr");
}
sycl::event ormrq(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, float *a, std::int64_t lda,
                  float *tau, float *c, std::int64_t ldc, float *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "ormrq");
}
sycl::event ormrq(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, double *a, std::int64_t lda,
                  double *tau, double *c, std::int64_t ldc, double *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "ormrq");
}
sycl::event ormqr(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, double *a, std::int64_t lda,
                  double *tau, double *c, std::int64_t ldc, double *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "ormqr");
}
sycl::event ormqr(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, float *a, std::int64_t lda,
                  float *tau, float *c, std::int64_t ldc, float *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "ormqr");
}

template <typename Func, typename T>
inline sycl::event potrf(Func func, sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                         T *a, std::int64_t lda, T *scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType *>(a);
            auto scratch_ = reinterpret_cast<cuDataType *>(scratchpad);
            cusolverStatus_t err;
            CUSOLVER_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(uplo), n, a_, lda, scratch_,
                                scratchpad_size, nullptr);
        });
    });
    return done;
}

#define POTRF_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                          \
    sycl::event potrf(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, TYPE *a,  \
                      std::int64_t lda, TYPE *scratchpad, std::int64_t scratchpad_size,     \
                      const std::vector<sycl::event> &dependencies) {                       \
        return potrf(CUSOLVER_ROUTINE, queue, uplo, n, a, lda, scratchpad, scratchpad_size, \
                     dependencies);                                                         \
    }

POTRF_LAUNCHER_USM(float, cusolverDnSpotrf)
POTRF_LAUNCHER_USM(double, cusolverDnDpotrf)
POTRF_LAUNCHER_USM(std::complex<float>, cusolverDnCpotrf)
POTRF_LAUNCHER_USM(std::complex<double>, cusolverDnZpotrf)

#undef POTRF_LAUNCHER_USM

sycl::event potri(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float *a,
                  std::int64_t lda, float *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "potri");
}
sycl::event potri(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double *a,
                  std::int64_t lda, double *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "potri");
}
sycl::event potri(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                  std::complex<float> *a, std::int64_t lda, std::complex<float> *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "potri");
}
sycl::event potri(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                  std::complex<double> *a, std::int64_t lda, std::complex<double> *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "potri");
}
sycl::event potrs(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
                  float *a, std::int64_t lda, float *b, std::int64_t ldb, float *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "potrs");
}
sycl::event potrs(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
                  double *a, std::int64_t lda, double *b, std::int64_t ldb, double *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "potrs");
}
sycl::event potrs(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
                  std::complex<float> *a, std::int64_t lda, std::complex<float> *b,
                  std::int64_t ldb, std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "potrs");
}
sycl::event potrs(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
                  std::complex<double> *a, std::int64_t lda, std::complex<double> *b,
                  std::int64_t ldb, std::complex<double> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "potrs");
}
sycl::event syevd(sycl::queue &queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                  double *a, std::int64_t lda, double *w, double *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "syevd");
}
sycl::event syevd(sycl::queue &queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                  float *a, std::int64_t lda, float *w, float *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "syevd");
}
sycl::event sygvd(sycl::queue &queue, std::int64_t itype, oneapi::mkl::job jobz,
                  oneapi::mkl::uplo uplo, std::int64_t n, double *a, std::int64_t lda, double *b,
                  std::int64_t ldb, double *w, double *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "sygvd");
}
sycl::event sygvd(sycl::queue &queue, std::int64_t itype, oneapi::mkl::job jobz,
                  oneapi::mkl::uplo uplo, std::int64_t n, float *a, std::int64_t lda, float *b,
                  std::int64_t ldb, float *w, float *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "sygvd");
}
sycl::event sytrd(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double *a,
                  std::int64_t lda, double *d, double *e, double *tau, double *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "sytrd");
}
sycl::event sytrd(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float *a,
                  std::int64_t lda, float *d, float *e, float *tau, float *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "sytrd");
}
sycl::event sytrf(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float *a,
                  std::int64_t lda, std::int64_t *ipiv, float *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "sytrf");
}
sycl::event sytrf(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double *a,
                  std::int64_t lda, std::int64_t *ipiv, double *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "sytrf");
}
sycl::event sytrf(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                  std::complex<float> *a, std::int64_t lda, std::int64_t *ipiv,
                  std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "sytrf");
}
sycl::event sytrf(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                  std::complex<double> *a, std::int64_t lda, std::int64_t *ipiv,
                  std::complex<double> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "sytrf");
}
sycl::event trtrs(sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                  oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs, std::complex<float> *a,
                  std::int64_t lda, std::complex<float> *b, std::int64_t ldb,
                  std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "trtrs");
}
sycl::event trtrs(sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                  oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs, double *a,
                  std::int64_t lda, double *b, std::int64_t ldb, double *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "trtrs");
}
sycl::event trtrs(sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                  oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs, float *a,
                  std::int64_t lda, float *b, std::int64_t ldb, float *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "trtrs");
}
sycl::event trtrs(sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                  oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs,
                  std::complex<double> *a, std::int64_t lda, std::complex<double> *b,
                  std::int64_t ldb, std::complex<double> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "trtrs");
}
sycl::event ungbr(sycl::queue &queue, oneapi::mkl::generate vec, std::int64_t m, std::int64_t n,
                  std::int64_t k, std::complex<float> *a, std::int64_t lda,
                  std::complex<float> *tau, std::complex<float> *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "ungbr");
}
sycl::event ungbr(sycl::queue &queue, oneapi::mkl::generate vec, std::int64_t m, std::int64_t n,
                  std::int64_t k, std::complex<double> *a, std::int64_t lda,
                  std::complex<double> *tau, std::complex<double> *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "ungbr");
}
sycl::event ungqr(sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k,
                  std::complex<float> *a, std::int64_t lda, std::complex<float> *tau,
                  std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "ungqr");
}
sycl::event ungqr(sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k,
                  std::complex<double> *a, std::int64_t lda, std::complex<double> *tau,
                  std::complex<double> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "ungqr");
}
sycl::event ungtr(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                  std::complex<float> *a, std::int64_t lda, std::complex<float> *tau,
                  std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "ungtr");
}
sycl::event ungtr(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                  std::complex<double> *a, std::int64_t lda, std::complex<double> *tau,
                  std::complex<double> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "ungtr");
}
sycl::event unmrq(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> *a,
                  std::int64_t lda, std::complex<float> *tau, std::complex<float> *c,
                  std::int64_t ldc, std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "unmrq");
}
sycl::event unmrq(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> *a,
                  std::int64_t lda, std::complex<double> *tau, std::complex<double> *c,
                  std::int64_t ldc, std::complex<double> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "unmrq");
}
sycl::event unmqr(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> *a,
                  std::int64_t lda, std::complex<float> *tau, std::complex<float> *c,
                  std::int64_t ldc, std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "unmqr");
}
sycl::event unmqr(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> *a,
                  std::int64_t lda, std::complex<double> *tau, std::complex<double> *c,
                  std::int64_t ldc, std::complex<double> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "unmqr");
}
sycl::event unmtr(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                  oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                  std::complex<float> *a, std::int64_t lda, std::complex<float> *tau,
                  std::complex<float> *c, std::int64_t ldc, std::complex<float> *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "unmtr");
}
sycl::event unmtr(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                  oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                  std::complex<double> *a, std::int64_t lda, std::complex<double> *tau,
                  std::complex<double> *c, std::int64_t ldc, std::complex<double> *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "unmtr");
}

// SCRATCHPAD APIs

template <typename Func>
inline void gebrd_scratchpad_size(Func func, sycl::queue &queue, std::int64_t m, std::int64_t n,
                                  std::int64_t lda, int *scratch_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            cusolverStatus_t err;
            CUSOLVER_ERROR_FUNC(func, err, handle, m, n, scratch_size);
        });
    });
}

#define GEBRD_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                            \
    template <>                                                                                   \
    std::int64_t gebrd_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t m, std::int64_t n, \
                                             std::int64_t lda) {                                  \
        int scratch_size;                                                                         \
        gebrd_scratchpad_size(CUSOLVER_ROUTINE, queue, m, n, lda, &scratch_size);                 \
        return scratch_size;                                                                      \
    }

GEBRD_LAUNCHER_SCRATCH(float, cusolverDnSgebrd_bufferSize)
GEBRD_LAUNCHER_SCRATCH(double, cusolverDnDgebrd_bufferSize)
GEBRD_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCgebrd_bufferSize)
GEBRD_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZgebrd_bufferSize)

#undef GEBRD_LAUNCHER_SCRATCH

template <>
std::int64_t gerqf_scratchpad_size<float>(sycl::queue &queue, std::int64_t m, std::int64_t n,
                                          std::int64_t lda) {
    throw unimplemented("lapack", "gerqf_scratchpad_size");
}
template <>
std::int64_t gerqf_scratchpad_size<double>(sycl::queue &queue, std::int64_t m, std::int64_t n,
                                           std::int64_t lda) {
    throw unimplemented("lapack", "gerqf_scratchpad_size");
}
template <>
std::int64_t gerqf_scratchpad_size<std::complex<float>>(sycl::queue &queue, std::int64_t m,
                                                        std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "gerqf_scratchpad_size");
}
template <>
std::int64_t gerqf_scratchpad_size<std::complex<double>>(sycl::queue &queue, std::int64_t m,
                                                         std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "gerqf_scratchpad_size");
}

template <typename Func>
inline void geqrf_scratchpad_size(Func func, sycl::queue &queue, std::int64_t m, std::int64_t n,
                                  std::int64_t lda, int *scratch_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            cusolverStatus_t err;
            CUSOLVER_ERROR_FUNC(func, err, handle, m, n, nullptr, lda, scratch_size);
        });
    });
}

#define GEQRF_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                            \
    template <>                                                                                   \
    std::int64_t geqrf_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t m, std::int64_t n, \
                                             std::int64_t lda) {                                  \
        int scratch_size;                                                                         \
        geqrf_scratchpad_size(CUSOLVER_ROUTINE, queue, m, n, lda, &scratch_size);                 \
        return scratch_size;                                                                      \
    }

GEQRF_LAUNCHER_SCRATCH(float, cusolverDnSgeqrf_bufferSize)
GEQRF_LAUNCHER_SCRATCH(double, cusolverDnDgeqrf_bufferSize)
GEQRF_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCgeqrf_bufferSize)
GEQRF_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZgeqrf_bufferSize)

#undef GEQRF_LAUNCHER_SCRATCH

template <typename Func>
inline void gesvd_scratchpad_size(Func func, sycl::queue &queue, oneapi::mkl::jobsvd jobu,
                                  oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n,
                                  std::int64_t lda, std::int64_t ldu, std::int64_t ldvt,
                                  int *scratch_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            cusolverStatus_t err;
            CUSOLVER_ERROR_FUNC(func, err, handle, m, n, scratch_size);
        });
    });
}

#define GESVD_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                            \
    template <>                                                                                   \
    std::int64_t gesvd_scratchpad_size<TYPE>(                                                     \
        sycl::queue & queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, std::int64_t m, \
        std::int64_t n, std::int64_t lda, std::int64_t ldu, std::int64_t ldvt) {                  \
        int scratch_size;                                                                         \
        gesvd_scratchpad_size(CUSOLVER_ROUTINE, queue, jobu, jobvt, m, n, lda, ldu, ldvt,         \
                              &scratch_size);                                                     \
        return scratch_size;                                                                      \
    }

GESVD_LAUNCHER_SCRATCH(float, cusolverDnSgesvd_bufferSize)
GESVD_LAUNCHER_SCRATCH(double, cusolverDnDgesvd_bufferSize)
GESVD_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCgesvd_bufferSize)
GESVD_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZgesvd_bufferSize)

#undef GESVD_LAUNCHER_SCRATCH

template <typename Func>
inline void getrf_scratchpad_size(Func func, sycl::queue &queue, std::int64_t m, std::int64_t n,
                                  std::int64_t lda, int *scratch_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            cusolverStatus_t err;
            CUSOLVER_ERROR_FUNC(func, err, handle, m, n, nullptr, lda, scratch_size);
        });
    });
}

#define GETRF_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                            \
    template <>                                                                                   \
    std::int64_t getrf_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t m, std::int64_t n, \
                                             std::int64_t lda) {                                  \
        int scratch_size;                                                                         \
        getrf_scratchpad_size(CUSOLVER_ROUTINE, queue, m, n, lda, &scratch_size);                 \
        return scratch_size;                                                                      \
    }

GETRF_LAUNCHER_SCRATCH(float, cusolverDnSgetrf_bufferSize)
GETRF_LAUNCHER_SCRATCH(double, cusolverDnDgetrf_bufferSize)
GETRF_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCgetrf_bufferSize)
GETRF_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZgetrf_bufferSize)

#undef GETRF_LAUNCHER_SCRATCH

template <>
std::int64_t getri_scratchpad_size<float>(sycl::queue &queue, std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "getri_scratchpad_size");
}
template <>
std::int64_t getri_scratchpad_size<double>(sycl::queue &queue, std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "getri_scratchpad_size");
}
template <>
std::int64_t getri_scratchpad_size<std::complex<float>>(sycl::queue &queue, std::int64_t n,
                                                        std::int64_t lda) {
    throw unimplemented("lapack", "getri_scratchpad_size");
}
template <>
std::int64_t getri_scratchpad_size<std::complex<double>>(sycl::queue &queue, std::int64_t n,
                                                         std::int64_t lda) {
    throw unimplemented("lapack", "getri_scratchpad_size");
}
// cusolverDnXgetrs does not use scratchpad memory
template <>
std::int64_t getrs_scratchpad_size<float>(sycl::queue &queue, oneapi::mkl::transpose trans,
                                          std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                          std::int64_t ldb) {
    return 0;
}
template <>
std::int64_t getrs_scratchpad_size<double>(sycl::queue &queue, oneapi::mkl::transpose trans,
                                           std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                           std::int64_t ldb) {
    return 0;
}
template <>
std::int64_t getrs_scratchpad_size<std::complex<float>>(sycl::queue &queue,
                                                        oneapi::mkl::transpose trans,
                                                        std::int64_t n, std::int64_t nrhs,
                                                        std::int64_t lda, std::int64_t ldb) {
    return 0;
}
template <>
std::int64_t getrs_scratchpad_size<std::complex<double>>(sycl::queue &queue,
                                                         oneapi::mkl::transpose trans,
                                                         std::int64_t n, std::int64_t nrhs,
                                                         std::int64_t lda, std::int64_t ldb) {
    return 0;
}
template <>
std::int64_t heevd_scratchpad_size<std::complex<float>>(sycl::queue &queue, oneapi::mkl::job jobz,
                                                        oneapi::mkl::uplo uplo, std::int64_t n,
                                                        std::int64_t lda) {
    throw unimplemented("lapack", "heevd_scratchpad_size");
}
template <>
std::int64_t heevd_scratchpad_size<std::complex<double>>(sycl::queue &queue, oneapi::mkl::job jobz,
                                                         oneapi::mkl::uplo uplo, std::int64_t n,
                                                         std::int64_t lda) {
    throw unimplemented("lapack", "heevd_scratchpad_size");
}
template <>
std::int64_t hegvd_scratchpad_size<std::complex<float>>(sycl::queue &queue, std::int64_t itype,
                                                        oneapi::mkl::job jobz,
                                                        oneapi::mkl::uplo uplo, std::int64_t n,
                                                        std::int64_t lda, std::int64_t ldb) {
    throw unimplemented("lapack", "hegvd_scratchpad_size");
}
template <>
std::int64_t hegvd_scratchpad_size<std::complex<double>>(sycl::queue &queue, std::int64_t itype,
                                                         oneapi::mkl::job jobz,
                                                         oneapi::mkl::uplo uplo, std::int64_t n,
                                                         std::int64_t lda, std::int64_t ldb) {
    throw unimplemented("lapack", "hegvd_scratchpad_size");
}
template <>
std::int64_t hetrd_scratchpad_size<std::complex<float>>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                                        std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "hetrd_scratchpad_size");
}
template <>
std::int64_t hetrd_scratchpad_size<std::complex<double>>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                                         std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "hetrd_scratchpad_size");
}
template <>
std::int64_t hetrf_scratchpad_size<std::complex<float>>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                                        std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "hetrf_scratchpad_size");
}
template <>
std::int64_t hetrf_scratchpad_size<std::complex<double>>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                                         std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "hetrf_scratchpad_size");
}
template <>
std::int64_t orgbr_scratchpad_size<float>(sycl::queue &queue, oneapi::mkl::generate vect,
                                          std::int64_t m, std::int64_t n, std::int64_t k,
                                          std::int64_t lda) {
    throw unimplemented("lapack", "orgbr_scratchpad_size");
}
template <>
std::int64_t orgbr_scratchpad_size<double>(sycl::queue &queue, oneapi::mkl::generate vect,
                                           std::int64_t m, std::int64_t n, std::int64_t k,
                                           std::int64_t lda) {
    throw unimplemented("lapack", "orgbr_scratchpad_size");
}
template <>
std::int64_t orgtr_scratchpad_size<float>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                          std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "orgtr_scratchpad_size");
}
template <>
std::int64_t orgtr_scratchpad_size<double>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                           std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "orgtr_scratchpad_size");
}
template <>
std::int64_t orgqr_scratchpad_size<float>(sycl::queue &queue, std::int64_t m, std::int64_t n,
                                          std::int64_t k, std::int64_t lda) {
    throw unimplemented("lapack", "orgqr_scratchpad_size");
}
template <>
std::int64_t orgqr_scratchpad_size<double>(sycl::queue &queue, std::int64_t m, std::int64_t n,
                                           std::int64_t k, std::int64_t lda) {
    throw unimplemented("lapack", "orgqr_scratchpad_size");
}
template <>
std::int64_t ormrq_scratchpad_size<float>(sycl::queue &queue, oneapi::mkl::side side,
                                          oneapi::mkl::transpose trans, std::int64_t m,
                                          std::int64_t n, std::int64_t k, std::int64_t lda,
                                          std::int64_t ldc) {
    throw unimplemented("lapack", "ormrq_scratchpad_size");
}
template <>
std::int64_t ormrq_scratchpad_size<double>(sycl::queue &queue, oneapi::mkl::side side,
                                           oneapi::mkl::transpose trans, std::int64_t m,
                                           std::int64_t n, std::int64_t k, std::int64_t lda,
                                           std::int64_t ldc) {
    throw unimplemented("lapack", "ormrq_scratchpad_size");
}
template <>
std::int64_t ormqr_scratchpad_size<float>(sycl::queue &queue, oneapi::mkl::side side,
                                          oneapi::mkl::transpose trans, std::int64_t m,
                                          std::int64_t n, std::int64_t k, std::int64_t lda,
                                          std::int64_t ldc) {
    throw unimplemented("lapack", "ormqr_scratchpad_size");
}
template <>
std::int64_t ormqr_scratchpad_size<double>(sycl::queue &queue, oneapi::mkl::side side,
                                           oneapi::mkl::transpose trans, std::int64_t m,
                                           std::int64_t n, std::int64_t k, std::int64_t lda,
                                           std::int64_t ldc) {
    throw unimplemented("lapack", "ormqr_scratchpad_size");
}
template <>
std::int64_t ormtr_scratchpad_size<float>(sycl::queue &queue, oneapi::mkl::side side,
                                          oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                                          std::int64_t m, std::int64_t n, std::int64_t lda,
                                          std::int64_t ldc) {
    throw unimplemented("lapack", "ormtr_scratchpad_size");
}
template <>
std::int64_t ormtr_scratchpad_size<double>(sycl::queue &queue, oneapi::mkl::side side,
                                           oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                                           std::int64_t m, std::int64_t n, std::int64_t lda,
                                           std::int64_t ldc) {
    throw unimplemented("lapack", "ormtr_scratchpad_size");
}

template <typename Func>
inline void potrf_scratchpad_size(Func func, sycl::queue &queue, oneapi::mkl::uplo uplo,
                                  std::int64_t n, std::int64_t lda, int *scratch_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            cusolverStatus_t err;
            CUSOLVER_ERROR_FUNC(func, err, handle, get_cublas_fill_mode(uplo), n, nullptr, lda,
                                scratch_size);
        });
    });
}

#define POTRF_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                    \
    template <>                                                                           \
    std::int64_t potrf_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::mkl::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {          \
        int scratch_size;                                                                 \
        potrf_scratchpad_size(CUSOLVER_ROUTINE, queue, uplo, n, lda, &scratch_size);      \
        return scratch_size;                                                              \
    }

POTRF_LAUNCHER_SCRATCH(float, cusolverDnSpotrf_bufferSize)
POTRF_LAUNCHER_SCRATCH(double, cusolverDnDpotrf_bufferSize)
POTRF_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCpotrf_bufferSize)
POTRF_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZpotrf_bufferSize)

#undef POTRF_LAUNCHER_SCRATCH

template <>
std::int64_t potrs_scratchpad_size<float>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                          std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                          std::int64_t ldb) {
    throw unimplemented("lapack", "potrs_scratchpad_size");
}
template <>
std::int64_t potrs_scratchpad_size<double>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                           std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                           std::int64_t ldb) {
    throw unimplemented("lapack", "potrs_scratchpad_size");
}
template <>
std::int64_t potrs_scratchpad_size<std::complex<float>>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                                        std::int64_t n, std::int64_t nrhs,
                                                        std::int64_t lda, std::int64_t ldb) {
    throw unimplemented("lapack", "potrs_scratchpad_size");
}
template <>
std::int64_t potrs_scratchpad_size<std::complex<double>>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                                         std::int64_t n, std::int64_t nrhs,
                                                         std::int64_t lda, std::int64_t ldb) {
    throw unimplemented("lapack", "potrs_scratchpad_size");
}
template <>
std::int64_t potri_scratchpad_size<float>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                          std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "potri_scratchpad_size");
}
template <>
std::int64_t potri_scratchpad_size<double>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                           std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "potri_scratchpad_size");
}
template <>
std::int64_t potri_scratchpad_size<std::complex<float>>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                                        std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "potri_scratchpad_size");
}
template <>
std::int64_t potri_scratchpad_size<std::complex<double>>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                                         std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "potri_scratchpad_size");
}
template <>
std::int64_t sytrf_scratchpad_size<float>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                          std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "sytrf_scratchpad_size");
}
template <>
std::int64_t sytrf_scratchpad_size<double>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                           std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "sytrf_scratchpad_size");
}
template <>
std::int64_t sytrf_scratchpad_size<std::complex<float>>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                                        std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "sytrf_scratchpad_size");
}
template <>
std::int64_t sytrf_scratchpad_size<std::complex<double>>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                                         std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "sytrf_scratchpad_size");
}
template <>
std::int64_t syevd_scratchpad_size<float>(sycl::queue &queue, oneapi::mkl::job jobz,
                                          oneapi::mkl::uplo uplo, std::int64_t n,
                                          std::int64_t lda) {
    throw unimplemented("lapack", "syevd_scratchpad_size");
}
template <>
std::int64_t syevd_scratchpad_size<double>(sycl::queue &queue, oneapi::mkl::job jobz,
                                           oneapi::mkl::uplo uplo, std::int64_t n,
                                           std::int64_t lda) {
    throw unimplemented("lapack", "syevd_scratchpad_size");
}
template <>
std::int64_t sygvd_scratchpad_size<float>(sycl::queue &queue, std::int64_t itype,
                                          oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                                          std::int64_t n, std::int64_t lda, std::int64_t ldb) {
    throw unimplemented("lapack", "sygvd_scratchpad_size");
}
template <>
std::int64_t sygvd_scratchpad_size<double>(sycl::queue &queue, std::int64_t itype,
                                           oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                                           std::int64_t n, std::int64_t lda, std::int64_t ldb) {
    throw unimplemented("lapack", "sygvd_scratchpad_size");
}
template <>
std::int64_t sytrd_scratchpad_size<float>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                          std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "sytrd_scratchpad_size");
}
template <>
std::int64_t sytrd_scratchpad_size<double>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                           std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "sytrd_scratchpad_size");
}
template <>
std::int64_t trtrs_scratchpad_size<float>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                          oneapi::mkl::transpose trans, oneapi::mkl::diag diag,
                                          std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                          std::int64_t ldb) {
    throw unimplemented("lapack", "trtrs_scratchpad_size");
}
template <>
std::int64_t trtrs_scratchpad_size<double>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                           oneapi::mkl::transpose trans, oneapi::mkl::diag diag,
                                           std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                           std::int64_t ldb) {
    throw unimplemented("lapack", "trtrs_scratchpad_size");
}
template <>
std::int64_t trtrs_scratchpad_size<std::complex<float>>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                                        oneapi::mkl::transpose trans,
                                                        oneapi::mkl::diag diag, std::int64_t n,
                                                        std::int64_t nrhs, std::int64_t lda,
                                                        std::int64_t ldb) {
    throw unimplemented("lapack", "trtrs_scratchpad_size");
}
template <>
std::int64_t trtrs_scratchpad_size<std::complex<double>>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                                         oneapi::mkl::transpose trans,
                                                         oneapi::mkl::diag diag, std::int64_t n,
                                                         std::int64_t nrhs, std::int64_t lda,
                                                         std::int64_t ldb) {
    throw unimplemented("lapack", "trtrs_scratchpad_size");
}
template <>
std::int64_t ungbr_scratchpad_size<std::complex<float>>(sycl::queue &queue,
                                                        oneapi::mkl::generate vect, std::int64_t m,
                                                        std::int64_t n, std::int64_t k,
                                                        std::int64_t lda) {
    throw unimplemented("lapack", "ungbr_scratchpad_size");
}
template <>
std::int64_t ungbr_scratchpad_size<std::complex<double>>(sycl::queue &queue,
                                                         oneapi::mkl::generate vect, std::int64_t m,
                                                         std::int64_t n, std::int64_t k,
                                                         std::int64_t lda) {
    throw unimplemented("lapack", "ungbr_scratchpad_size");
}
template <>
std::int64_t ungqr_scratchpad_size<std::complex<float>>(sycl::queue &queue, std::int64_t m,
                                                        std::int64_t n, std::int64_t k,
                                                        std::int64_t lda) {
    throw unimplemented("lapack", "ungqr_scratchpad_size");
}
template <>
std::int64_t ungqr_scratchpad_size<std::complex<double>>(sycl::queue &queue, std::int64_t m,
                                                         std::int64_t n, std::int64_t k,
                                                         std::int64_t lda) {
    throw unimplemented("lapack", "ungqr_scratchpad_size");
}
template <>
std::int64_t ungtr_scratchpad_size<std::complex<float>>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                                        std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "ungtr_scratchpad_size");
}
template <>
std::int64_t ungtr_scratchpad_size<std::complex<double>>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                                         std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "ungtr_scratchpad_size");
}
template <>
std::int64_t unmrq_scratchpad_size<std::complex<float>>(sycl::queue &queue, oneapi::mkl::side side,
                                                        oneapi::mkl::transpose trans,
                                                        std::int64_t m, std::int64_t n,
                                                        std::int64_t k, std::int64_t lda,
                                                        std::int64_t ldc) {
    throw unimplemented("lapack", "unmrq_scratchpad_size");
}
template <>
std::int64_t unmrq_scratchpad_size<std::complex<double>>(sycl::queue &queue, oneapi::mkl::side side,
                                                         oneapi::mkl::transpose trans,
                                                         std::int64_t m, std::int64_t n,
                                                         std::int64_t k, std::int64_t lda,
                                                         std::int64_t ldc) {
    throw unimplemented("lapack", "unmrq_scratchpad_size");
}
template <>
std::int64_t unmqr_scratchpad_size<std::complex<float>>(sycl::queue &queue, oneapi::mkl::side side,
                                                        oneapi::mkl::transpose trans,
                                                        std::int64_t m, std::int64_t n,
                                                        std::int64_t k, std::int64_t lda,
                                                        std::int64_t ldc) {
    throw unimplemented("lapack", "unmqr_scratchpad_size");
}
template <>
std::int64_t unmqr_scratchpad_size<std::complex<double>>(sycl::queue &queue, oneapi::mkl::side side,
                                                         oneapi::mkl::transpose trans,
                                                         std::int64_t m, std::int64_t n,
                                                         std::int64_t k, std::int64_t lda,
                                                         std::int64_t ldc) {
    throw unimplemented("lapack", "unmqr_scratchpad_size");
}
template <>
std::int64_t unmtr_scratchpad_size<std::complex<float>>(sycl::queue &queue, oneapi::mkl::side side,
                                                        oneapi::mkl::uplo uplo,
                                                        oneapi::mkl::transpose trans,
                                                        std::int64_t m, std::int64_t n,
                                                        std::int64_t lda, std::int64_t ldc) {
    throw unimplemented("lapack", "unmtr_scratchpad_size");
}
template <>
std::int64_t unmtr_scratchpad_size<std::complex<double>>(sycl::queue &queue, oneapi::mkl::side side,
                                                         oneapi::mkl::uplo uplo,
                                                         oneapi::mkl::transpose trans,
                                                         std::int64_t m, std::int64_t n,
                                                         std::int64_t lda, std::int64_t ldc) {
    throw unimplemented("lapack", "unmtr_scratchpad_size");
}

} // namespace cusolver
} // namespace lapack
} // namespace mkl
} // namespace oneapi
