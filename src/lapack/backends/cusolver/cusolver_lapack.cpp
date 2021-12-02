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

void gebrd(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<std::complex<float>> &a,
           std::int64_t lda, sycl::buffer<float> &d, sycl::buffer<float> &e,
           sycl::buffer<std::complex<float>> &tauq, sycl::buffer<std::complex<float>> &taup,
           sycl::buffer<std::complex<float>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gebrd");
}
void gebrd(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<double> &a,
           std::int64_t lda, sycl::buffer<double> &d, sycl::buffer<double> &e,
           sycl::buffer<double> &tauq, sycl::buffer<double> &taup, sycl::buffer<double> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gebrd");
}
void gebrd(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<float> &a,
           std::int64_t lda, sycl::buffer<float> &d, sycl::buffer<float> &e,
           sycl::buffer<float> &tauq, sycl::buffer<float> &taup, sycl::buffer<float> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gebrd");
}
void gebrd(sycl::queue &queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>> &a, std::int64_t lda, sycl::buffer<double> &d,
           sycl::buffer<double> &e, sycl::buffer<std::complex<double>> &tauq,
           sycl::buffer<std::complex<double>> &taup, sycl::buffer<std::complex<double>> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gebrd");
}
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
void geqrf(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<std::complex<float>> &a,
           std::int64_t lda, sycl::buffer<std::complex<float>> &tau,
           sycl::buffer<std::complex<float>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "geqrf");
}
void geqrf(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<double> &a,
           std::int64_t lda, sycl::buffer<double> &tau, sycl::buffer<double> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "geqrf");
}
void geqrf(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<float> &a,
           std::int64_t lda, sycl::buffer<float> &tau, sycl::buffer<float> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "geqrf");
}
void geqrf(sycl::queue &queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>> &tau, sycl::buffer<std::complex<double>> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "geqrf");
}
void getrf(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<std::complex<float>> &a,
           std::int64_t lda, sycl::buffer<std::int64_t> &ipiv,
           sycl::buffer<std::complex<float>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrf");
}
void getrf(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<double> &a,
           std::int64_t lda, sycl::buffer<std::int64_t> &ipiv, sycl::buffer<double> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrf");
}
void getrf(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<float> &a,
           std::int64_t lda, sycl::buffer<std::int64_t> &ipiv, sycl::buffer<float> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrf");
}
void getrf(sycl::queue &queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>> &a, std::int64_t lda,
           sycl::buffer<std::int64_t> &ipiv, sycl::buffer<std::complex<double>> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrf");
}
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
void getrs(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<std::complex<float>> &a, std::int64_t lda, sycl::buffer<std::int64_t> &ipiv,
           sycl::buffer<std::complex<float>> &b, std::int64_t ldb,
           sycl::buffer<std::complex<float>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrs");
}
void getrs(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<double> &a, std::int64_t lda, sycl::buffer<std::int64_t> &ipiv,
           sycl::buffer<double> &b, std::int64_t ldb, sycl::buffer<double> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrs");
}
void getrs(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<float> &a, std::int64_t lda, sycl::buffer<std::int64_t> &ipiv,
           sycl::buffer<float> &b, std::int64_t ldb, sycl::buffer<float> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrs");
}
void getrs(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<std::complex<double>> &a, std::int64_t lda,
           sycl::buffer<std::int64_t> &ipiv, sycl::buffer<std::complex<double>> &b,
           std::int64_t ldb, sycl::buffer<std::complex<double>> &scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrs");
}
void gesvd(sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, std::int64_t m,
           std::int64_t n, sycl::buffer<double> &a, std::int64_t lda, sycl::buffer<double> &s,
           sycl::buffer<double> &u, std::int64_t ldu, sycl::buffer<double> &vt, std::int64_t ldvt,
           sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gesvd");
}
void gesvd(sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, std::int64_t m,
           std::int64_t n, sycl::buffer<float> &a, std::int64_t lda, sycl::buffer<float> &s,
           sycl::buffer<float> &u, std::int64_t ldu, sycl::buffer<float> &vt, std::int64_t ldvt,
           sycl::buffer<float> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gesvd");
}
void gesvd(sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, std::int64_t m,
           std::int64_t n, sycl::buffer<std::complex<float>> &a, std::int64_t lda,
           sycl::buffer<float> &s, sycl::buffer<std::complex<float>> &u, std::int64_t ldu,
           sycl::buffer<std::complex<float>> &vt, std::int64_t ldvt,
           sycl::buffer<std::complex<float>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gesvd");
}
void gesvd(sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, std::int64_t m,
           std::int64_t n, sycl::buffer<std::complex<double>> &a, std::int64_t lda,
           sycl::buffer<double> &s, sycl::buffer<std::complex<double>> &u, std::int64_t ldu,
           sycl::buffer<std::complex<double>> &vt, std::int64_t ldvt,
           sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gesvd");
}
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

sycl::event gebrd(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> *a,
                  std::int64_t lda, float *d, float *e, std::complex<float> *tauq,
                  std::complex<float> *taup, std::complex<float> *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "gebrd");
}
sycl::event gebrd(sycl::queue &queue, std::int64_t m, std::int64_t n, double *a, std::int64_t lda,
                  double *d, double *e, double *tauq, double *taup, double *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "gebrd");
}
sycl::event gebrd(sycl::queue &queue, std::int64_t m, std::int64_t n, float *a, std::int64_t lda,
                  float *d, float *e, float *tauq, float *taup, float *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "gebrd");
}
sycl::event gebrd(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> *a,
                  std::int64_t lda, double *d, double *e, std::complex<double> *tauq,
                  std::complex<double> *taup, std::complex<double> *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "gebrd");
}
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
sycl::event geqrf(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> *a,
                  std::int64_t lda, std::complex<float> *tau, std::complex<float> *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "geqrf");
}
sycl::event geqrf(sycl::queue &queue, std::int64_t m, std::int64_t n, double *a, std::int64_t lda,
                  double *tau, double *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "geqrf");
}
sycl::event geqrf(sycl::queue &queue, std::int64_t m, std::int64_t n, float *a, std::int64_t lda,
                  float *tau, float *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "geqrf");
}
sycl::event geqrf(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> *a,
                  std::int64_t lda, std::complex<double> *tau, std::complex<double> *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "geqrf");
}
sycl::event getrf(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> *a,
                  std::int64_t lda, std::int64_t *ipiv, std::complex<float> *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getrf");
}
sycl::event getrf(sycl::queue &queue, std::int64_t m, std::int64_t n, double *a, std::int64_t lda,
                  std::int64_t *ipiv, double *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getrf");
}
sycl::event getrf(sycl::queue &queue, std::int64_t m, std::int64_t n, float *a, std::int64_t lda,
                  std::int64_t *ipiv, float *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getrf");
}
sycl::event getrf(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> *a,
                  std::int64_t lda, std::int64_t *ipiv, std::complex<double> *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getrf");
}
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
sycl::event getrs(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n,
                  std::int64_t nrhs, std::complex<float> *a, std::int64_t lda, std::int64_t *ipiv,
                  std::complex<float> *b, std::int64_t ldb, std::complex<float> *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getrs");
}
sycl::event getrs(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n,
                  std::int64_t nrhs, double *a, std::int64_t lda, std::int64_t *ipiv, double *b,
                  std::int64_t ldb, double *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getrs");
}
sycl::event getrs(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n,
                  std::int64_t nrhs, float *a, std::int64_t lda, std::int64_t *ipiv, float *b,
                  std::int64_t ldb, float *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getrs");
}
sycl::event getrs(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n,
                  std::int64_t nrhs, std::complex<double> *a, std::int64_t lda, std::int64_t *ipiv,
                  std::complex<double> *b, std::int64_t ldb, std::complex<double> *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getrs");
}
sycl::event gesvd(sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt,
                  std::int64_t m, std::int64_t n, double *a, std::int64_t lda, double *s, double *u,
                  std::int64_t ldu, double *vt, std::int64_t ldvt, double *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "gesvd");
}
sycl::event gesvd(sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt,
                  std::int64_t m, std::int64_t n, float *a, std::int64_t lda, float *s, float *u,
                  std::int64_t ldu, float *vt, std::int64_t ldvt, float *scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "gesvd");
}
sycl::event gesvd(sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt,
                  std::int64_t m, std::int64_t n, std::complex<float> *a, std::int64_t lda,
                  float *s, std::complex<float> *u, std::int64_t ldu, std::complex<float> *vt,
                  std::int64_t ldvt, std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "gesvd");
}
sycl::event gesvd(sycl::queue &queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt,
                  std::int64_t m, std::int64_t n, std::complex<double> *a, std::int64_t lda,
                  double *s, std::complex<double> *u, std::int64_t ldu, std::complex<double> *vt,
                  std::int64_t ldvt, std::complex<double> *scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "gesvd");
}
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

template <>
std::int64_t gebrd_scratchpad_size<float>(sycl::queue &queue, std::int64_t m, std::int64_t n,
                                          std::int64_t lda) {
    throw unimplemented("lapack", "gebrd_scratchpad_size");
}
template <>
std::int64_t gebrd_scratchpad_size<double>(sycl::queue &queue, std::int64_t m, std::int64_t n,
                                           std::int64_t lda) {
    throw unimplemented("lapack", "gebrd_scratchpad_size");
}
template <>
std::int64_t gebrd_scratchpad_size<std::complex<float>>(sycl::queue &queue, std::int64_t m,
                                                        std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "gebrd_scratchpad_size");
}
template <>
std::int64_t gebrd_scratchpad_size<std::complex<double>>(sycl::queue &queue, std::int64_t m,
                                                         std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "gebrd_scratchpad_size");
}
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
template <>
std::int64_t geqrf_scratchpad_size<float>(sycl::queue &queue, std::int64_t m, std::int64_t n,
                                          std::int64_t lda) {
    throw unimplemented("lapack", "geqrf_scratchpad_size");
}
template <>
std::int64_t geqrf_scratchpad_size<double>(sycl::queue &queue, std::int64_t m, std::int64_t n,
                                           std::int64_t lda) {
    throw unimplemented("lapack", "geqrf_scratchpad_size");
}
template <>
std::int64_t geqrf_scratchpad_size<std::complex<float>>(sycl::queue &queue, std::int64_t m,
                                                        std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "geqrf_scratchpad_size");
}
template <>
std::int64_t geqrf_scratchpad_size<std::complex<double>>(sycl::queue &queue, std::int64_t m,
                                                         std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "geqrf_scratchpad_size");
}
template <>
std::int64_t gesvd_scratchpad_size<float>(sycl::queue &queue, oneapi::mkl::jobsvd jobu,
                                          oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n,
                                          std::int64_t lda, std::int64_t ldu, std::int64_t ldvt) {
    throw unimplemented("lapack", "gesvd_scratchpad_size");
}
template <>
std::int64_t gesvd_scratchpad_size<double>(sycl::queue &queue, oneapi::mkl::jobsvd jobu,
                                           oneapi::mkl::jobsvd jobvt, std::int64_t m,
                                           std::int64_t n, std::int64_t lda, std::int64_t ldu,
                                           std::int64_t ldvt) {
    throw unimplemented("lapack", "gesvd_scratchpad_size");
}
template <>
std::int64_t gesvd_scratchpad_size<std::complex<float>>(sycl::queue &queue,
                                                        oneapi::mkl::jobsvd jobu,
                                                        oneapi::mkl::jobsvd jobvt, std::int64_t m,
                                                        std::int64_t n, std::int64_t lda,
                                                        std::int64_t ldu, std::int64_t ldvt) {
    throw unimplemented("lapack", "gesvd_scratchpad_size");
}
template <>
std::int64_t gesvd_scratchpad_size<std::complex<double>>(sycl::queue &queue,
                                                         oneapi::mkl::jobsvd jobu,
                                                         oneapi::mkl::jobsvd jobvt, std::int64_t m,
                                                         std::int64_t n, std::int64_t lda,
                                                         std::int64_t ldu, std::int64_t ldvt) {
    throw unimplemented("lapack", "gesvd_scratchpad_size");
}
template <>
std::int64_t getrf_scratchpad_size<float>(sycl::queue &queue, std::int64_t m, std::int64_t n,
                                          std::int64_t lda) {
    throw unimplemented("lapack", "getrf_scratchpad_size");
}
template <>
std::int64_t getrf_scratchpad_size<double>(sycl::queue &queue, std::int64_t m, std::int64_t n,
                                           std::int64_t lda) {
    throw unimplemented("lapack", "getrf_scratchpad_size");
}
template <>
std::int64_t getrf_scratchpad_size<std::complex<float>>(sycl::queue &queue, std::int64_t m,
                                                        std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "getrf_scratchpad_size");
}
template <>
std::int64_t getrf_scratchpad_size<std::complex<double>>(sycl::queue &queue, std::int64_t m,
                                                         std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "getrf_scratchpad_size");
}
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
template <>
std::int64_t getrs_scratchpad_size<float>(sycl::queue &queue, oneapi::mkl::transpose trans,
                                          std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                          std::int64_t ldb) {
    throw unimplemented("lapack", "getrs_scratchpad_size");
}
template <>
std::int64_t getrs_scratchpad_size<double>(sycl::queue &queue, oneapi::mkl::transpose trans,
                                           std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                           std::int64_t ldb) {
    throw unimplemented("lapack", "getrs_scratchpad_size");
}
template <>
std::int64_t getrs_scratchpad_size<std::complex<float>>(sycl::queue &queue,
                                                        oneapi::mkl::transpose trans,
                                                        std::int64_t n, std::int64_t nrhs,
                                                        std::int64_t lda, std::int64_t ldb) {
    throw unimplemented("lapack", "getrs_scratchpad_size");
}
template <>
std::int64_t getrs_scratchpad_size<std::complex<double>>(sycl::queue &queue,
                                                         oneapi::mkl::transpose trans,
                                                         std::int64_t n, std::int64_t nrhs,
                                                         std::int64_t lda, std::int64_t ldb) {
    throw unimplemented("lapack", "getrs_scratchpad_size");
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
