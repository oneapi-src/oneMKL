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

// BATCH BUFFER API

void geqrf_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<float> &a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<float> &tau,
                 std::int64_t stride_tau, std::int64_t batch_size, sycl::buffer<float> &scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "geqrf_batch");
}
void geqrf_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<double> &a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<double> &tau,
                 std::int64_t stride_tau, std::int64_t batch_size, sycl::buffer<double> &scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "geqrf_batch");
}
void geqrf_batch(sycl::queue &queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<float>> &a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<float>> &tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>> &scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "geqrf_batch");
}
void geqrf_batch(sycl::queue &queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<double>> &a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<double>> &tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>> &scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "geqrf_batch");
}
void getri_batch(sycl::queue &queue, std::int64_t n, sycl::buffer<float> &a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<float> &scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri_batch");
}
void getri_batch(sycl::queue &queue, std::int64_t n, sycl::buffer<double> &a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<double> &scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri_batch");
}
void getri_batch(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>> &a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t> &ipiv,
                 std::int64_t stride_ipiv, std::int64_t batch_size,
                 sycl::buffer<std::complex<float>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri_batch");
}
void getri_batch(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>> &a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t> &ipiv,
                 std::int64_t stride_ipiv, std::int64_t batch_size,
                 sycl::buffer<std::complex<double>> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri_batch");
}

template <typename Func, typename T>
inline void getrs_batch(const char *Func_name, Func func, sycl::queue &queue,
                        oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
                        sycl::buffer<T> &a, std::int64_t lda, std::int64_t stride_a,
                        sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv,
                        sycl::buffer<T> &b, std::int64_t ldb, std::int64_t stride_b,
                        std::int64_t batch_size, sycl::buffer<T> &scratchpad,
                        std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(n, nrhs, lda, ldb, stride_ipiv, stride_b, batch_size, scratchpad_size);

    // cuSolver legacy api does not accept 64-bit ints.
    // To get around the limitation.
    // Create new buffer and convert 64-bit values.
    std::uint64_t ipiv_size = stride_ipiv * batch_size;
    sycl::buffer<int, 1> ipiv32(sycl::range<1>{ ipiv_size });

    queue.submit([&](sycl::handler &cgh) {
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::write>(cgh);
        auto ipiv_acc = ipiv.template get_access<sycl::access::mode::read>(cgh);
        cgh.parallel_for(sycl::range<1>{ ipiv_size }, [=](sycl::id<1> index) {
            ipiv32_acc[index] = static_cast<std::int32_t>(ipiv_acc[index]);
        });
    });

    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto ipiv_acc = ipiv32.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::write>(cgh);

        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType *>(a_acc);
            auto ipiv_ = sc.get_mem<std::int32_t *>(ipiv_acc);
            auto b_ = sc.get_mem<cuDataType *>(b_acc);

            for (int64_t i = 0; i < batch_size; ++i) {
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T(Func_name, func, err, handle, get_cublas_operation(trans), n,
                                      nrhs, a_ + stride_a * i, lda, ipiv_ + stride_ipiv * i,
                                      b_ + stride_b * i, ldb, nullptr);
            }
        });
    });
}

// Scratchpad memory not needed as parts of buffer a is used as workspace memory
#define GETRS_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                          \
    void getrs_batch(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n,            \
                     std::int64_t nrhs, sycl::buffer<TYPE> &a, std::int64_t lda,                  \
                     std::int64_t stride_a, sycl::buffer<std::int64_t> &ipiv,                     \
                     std::int64_t stride_ipiv, sycl::buffer<TYPE> &b, std::int64_t ldb,           \
                     std::int64_t stride_b, std::int64_t batch_size,                              \
                     sycl::buffer<TYPE> &scratchpad, std::int64_t scratchpad_size) {              \
        return getrs_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, trans, n, nrhs, a, lda,    \
                           stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size, scratchpad, \
                           scratchpad_size);                                                      \
    }

GETRS_BATCH_LAUNCHER_USM(float, cusolverDnSgetrs)
GETRS_BATCH_LAUNCHER_USM(double, cusolverDnDgetrs)
GETRS_BATCH_LAUNCHER_USM(std::complex<float>, cusolverDnCgetrs)
GETRS_BATCH_LAUNCHER_USM(std::complex<double>, cusolverDnZgetrs)

#undef GETRS_BATCH_LAUNCHER_USM

void getrf_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<float> &a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t> &ipiv,
                 std::int64_t stride_ipiv, std::int64_t batch_size, sycl::buffer<float> &scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrf_batch");
}
void getrf_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<double> &a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t> &ipiv,
                 std::int64_t stride_ipiv, std::int64_t batch_size,
                 sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrf_batch");
}
void getrf_batch(sycl::queue &queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<float>> &a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>> &scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrf_batch");
}
void getrf_batch(sycl::queue &queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<double>> &a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>> &scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrf_batch");
}
void orgqr_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k,
                 sycl::buffer<float> &a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<float> &tau, std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<float> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "orgqr_batch");
}
void orgqr_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k,
                 sycl::buffer<double> &a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<double> &tau, std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<double> &scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "orgqr_batch");
}

template <typename Func, typename T>
inline void potrf_batch(const char *func_name, Func func, sycl::queue &queue,
                        oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<T> &a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size,
                        sycl::buffer<T> &scratchpad, std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(n, lda, stride_a, batch_size, scratchpad_size);

    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            int64_t offset = 0;
            cusolverStatus_t err;
            CUresult cuda_result;
            CUdeviceptr a_dev;

            auto a_ = sc.get_mem<cuDataType *>(a_acc);

            // Transform ptr and stride to list of ptr's
            cuDataType **a_batched = create_ptr_list_from_stride(a_, stride_a, batch_size);
            CUDA_ERROR_FUNC(cuMemAlloc, cuda_result, &a_dev, sizeof(T *) * batch_size);
            CUDA_ERROR_FUNC(cuMemcpyHtoD, cuda_result, a_dev, a_batched, sizeof(T *) * batch_size);

            auto **a_dev_ = reinterpret_cast<cuDataType **>(a_dev);

            CUSOLVER_ERROR_FUNC_T(func_name, func, err, handle, get_cublas_fill_mode(uplo), (int)n,
                                  a_dev_, (int)lda, nullptr, (int)batch_size);

            cuMemFree(a_dev);
        });
    });
}

// Scratchpad memory not needed as parts of buffer a is used as workspace memory
#define POTRF_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                          \
    void potrf_batch(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,                  \
                     sycl::buffer<TYPE> &a, std::int64_t lda, std::int64_t stride_a,              \
                     std::int64_t batch_size, sycl::buffer<TYPE> &scratchpad,                     \
                     std::int64_t scratchpad_size) {                                              \
        return potrf_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda, stride_a, \
                           batch_size, scratchpad, scratchpad_size);                              \
    }

POTRF_BATCH_LAUNCHER_USM(float, cusolverDnSpotrfBatched)
POTRF_BATCH_LAUNCHER_USM(double, cusolverDnDpotrfBatched)
POTRF_BATCH_LAUNCHER_USM(std::complex<float>, cusolverDnCpotrfBatched)
POTRF_BATCH_LAUNCHER_USM(std::complex<double>, cusolverDnZpotrfBatched)

#undef POTRF_BATCH_LAUNCHER_USM

void potrs_batch(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
                 sycl::buffer<float> &a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<float> &b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<float> &scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrs_batch");
}
void potrs_batch(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
                 sycl::buffer<double> &a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<double> &b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<double> &scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrs_batch");
}
void potrs_batch(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
                 sycl::buffer<std::complex<float>> &a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<float>> &b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>> &scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrs_batch");
}
void potrs_batch(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
                 sycl::buffer<std::complex<double>> &a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<double>> &b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>> &scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrs_batch");
}
void ungqr_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k,
                 sycl::buffer<std::complex<float>> &a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<float>> &tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>> &scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ungqr_batch");
}
void ungqr_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k,
                 sycl::buffer<std::complex<double>> &a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<double>> &tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>> &scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ungqr_batch");
}

// BATCH USM API

sycl::event geqrf_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, float *a,
                        std::int64_t lda, std::int64_t stride_a, float *tau,
                        std::int64_t stride_tau, std::int64_t batch_size, float *scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, double *a,
                        std::int64_t lda, std::int64_t stride_a, double *tau,
                        std::int64_t stride_tau, std::int64_t batch_size, double *scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> *a,
                        std::int64_t lda, std::int64_t stride_a, std::complex<float> *tau,
                        std::int64_t stride_tau, std::int64_t batch_size,
                        std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> *a,
                        std::int64_t lda, std::int64_t stride_a, std::complex<double> *tau,
                        std::int64_t stride_tau, std::int64_t batch_size,
                        std::complex<double> *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue &queue, std::int64_t *m, std::int64_t *n, float **a,
                        std::int64_t *lda, float **tau, std::int64_t group_count,
                        std::int64_t *group_sizes, float *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue &queue, std::int64_t *m, std::int64_t *n, double **a,
                        std::int64_t *lda, double **tau, std::int64_t group_count,
                        std::int64_t *group_sizes, double *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue &queue, std::int64_t *m, std::int64_t *n,
                        std::complex<float> **a, std::int64_t *lda, std::complex<float> **tau,
                        std::int64_t group_count, std::int64_t *group_sizes,
                        std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue &queue, std::int64_t *m, std::int64_t *n,
                        std::complex<double> **a, std::int64_t *lda, std::complex<double> **tau,
                        std::int64_t group_count, std::int64_t *group_sizes,
                        std::complex<double> *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event getrf_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, float *a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size, float *scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, double *a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size, double *scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> *a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> *a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<double> *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue &queue, std::int64_t *m, std::int64_t *n, float **a,
                        std::int64_t *lda, std::int64_t **ipiv, std::int64_t group_count,
                        std::int64_t *group_sizes, float *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue &queue, std::int64_t *m, std::int64_t *n, double **a,
                        std::int64_t *lda, std::int64_t **ipiv, std::int64_t group_count,
                        std::int64_t *group_sizes, double *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue &queue, std::int64_t *m, std::int64_t *n,
                        std::complex<float> **a, std::int64_t *lda, std::int64_t **ipiv,
                        std::int64_t group_count, std::int64_t *group_sizes,
                        std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue &queue, std::int64_t *m, std::int64_t *n,
                        std::complex<double> **a, std::int64_t *lda, std::int64_t **ipiv,
                        std::int64_t group_count, std::int64_t *group_sizes,
                        std::complex<double> *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getri_batch(sycl::queue &queue, std::int64_t n, float *a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t *ipiv, std::int64_t stride_ipiv,
                        std::int64_t batch_size, float *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue &queue, std::int64_t n, double *a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t *ipiv, std::int64_t stride_ipiv,
                        std::int64_t batch_size, double *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue &queue, std::int64_t n, std::complex<float> *a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue &queue, std::int64_t n, std::complex<double> *a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<double> *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue &queue, std::int64_t *n, float **a, std::int64_t *lda,
                        std::int64_t **ipiv, std::int64_t group_count, std::int64_t *group_sizes,
                        float *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue &queue, std::int64_t *n, double **a, std::int64_t *lda,
                        std::int64_t **ipiv, std::int64_t group_count, std::int64_t *group_sizes,
                        double *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue &queue, std::int64_t *n, std::complex<float> **a,
                        std::int64_t *lda, std::int64_t **ipiv, std::int64_t group_count,
                        std::int64_t *group_sizes, std::complex<float> *scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue &queue, std::int64_t *n, std::complex<double> **a,
                        std::int64_t *lda, std::int64_t **ipiv, std::int64_t group_count,
                        std::int64_t *group_sizes, std::complex<double> *scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "getri_batch");
}

template <typename Func, typename T>
inline sycl::event getrs_batch(const char *Func_name, Func func, sycl::queue &queue,
                               oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
                               T *a, std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv,
                               std::int64_t stride_ipiv, T *b, std::int64_t ldb,
                               std::int64_t stride_b, std::int64_t batch_size, T *scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(n, nrhs, lda, ldb, stride_ipiv, stride_b, batch_size, scratchpad_size);

    // cuSolver legacy api does not accept 64-bit ints.
    // To get around the limitation.
    // Create new buffer and convert 64-bit values.
    std::uint64_t ipiv_size = stride_ipiv * batch_size;
    int *ipiv32 = (int *)malloc_device(sizeof(int) * ipiv_size, queue);

    auto done_casting = queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{ ipiv_size }, [=](sycl::id<1> index) {
            ipiv32[index] = static_cast<std::int32_t>(ipiv[index]);
        });
    });

    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        cgh.depends_on(done_casting);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType *>(a);
            auto ipiv_ = reinterpret_cast<int *>(ipiv32);
            auto b_ = reinterpret_cast<cuDataType *>(b);

            for (int64_t i = 0; i < batch_size; ++i) {
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T(Func_name, func, err, handle, get_cublas_operation(trans), n,
                                      nrhs, a_ + stride_a * i, lda, ipiv_ + stride_ipiv * i,
                                      b_ + stride_b * i, ldb, nullptr);
            }
        });

        sycl::free(ipiv32, queue);
    });

    return done;
}

// Scratchpad memory not needed as parts of buffer a is used as workspace memory
#define GETRS_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                          \
    sycl::event getrs_batch(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n,     \
                            std::int64_t nrhs, TYPE *a, std::int64_t lda, std::int64_t stride_a,  \
                            std::int64_t *ipiv, std::int64_t stride_ipiv, TYPE *b,                \
                            std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,     \
                            TYPE *scratchpad, std::int64_t scratchpad_size,                       \
                            const std::vector<sycl::event> &dependencies) {                       \
        return getrs_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, trans, n, nrhs, a, lda,    \
                           stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size, scratchpad, \
                           scratchpad_size, dependencies);                                        \
    }

GETRS_BATCH_LAUNCHER_USM(float, cusolverDnSgetrs)
GETRS_BATCH_LAUNCHER_USM(double, cusolverDnDgetrs)
GETRS_BATCH_LAUNCHER_USM(std::complex<float>, cusolverDnCgetrs)
GETRS_BATCH_LAUNCHER_USM(std::complex<double>, cusolverDnZgetrs)

#undef GETRS_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event getrs_batch(const char *Func_name, Func func, sycl::queue &queue,
                               oneapi::mkl::transpose *trans, std::int64_t *n, std::int64_t *nrhs,
                               T **a, std::int64_t *lda, std::int64_t **ipiv, T **b,
                               std::int64_t *ldb, std::int64_t group_count,
                               std::int64_t *group_sizes, T *scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    int64_t batch_size = 0;
    overflow_check(group_count, scratchpad_size);
    for (int64_t i = 0; i < group_count; ++i) {
        overflow_check(n[i], nrhs[i], lda[i], ldb[i], group_sizes[i]);
        batch_size += group_sizes[i];
    }

    // cuSolver legacy api does not accept 64-bit ints.
    // ipiv is an array of pointers in host memory, pointing to
    // an array of 64-bit ints in device memory. Each vec of ipiv
    // values need to be converted from 64-bit to 32-bit. The list
    // must stay on host.
    int **ipiv32 = (int **)malloc(sizeof(int *) * batch_size + 1);
    std::vector<sycl::event> casting_dependencies(batch_size);
    int64_t global_id = 0;
    for (int64_t group_id = 0; group_id < group_count; ++group_id) {
        for (int64_t local_id = 0; local_id < group_sizes[group_id]; ++local_id, ++global_id) {
            uint64_t ipiv_size = n[group_id];
            int *d_group_ipiv32 = (int *)malloc_device(sizeof(int) * ipiv_size, queue);
            ipiv32[global_id] = d_group_ipiv32;
            int64_t *d_group_ipiv = ipiv[global_id];

            auto e = queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::range<1>{ ipiv_size }, [=](sycl::id<1> index) {
                    d_group_ipiv32[index] = static_cast<std::int32_t>(d_group_ipiv[index]);
                });
            });
            casting_dependencies[global_id] = e;
        }
    }

    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        for (int64_t i = 0; i < batch_size; i++) {
            cgh.depends_on(casting_dependencies[i]);
        }

        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            cusolverStatus_t err;
            auto a_ = reinterpret_cast<cuDataType **>(a);
            auto b_ = reinterpret_cast<cuDataType **>(b);
            int64_t global_id = 0;

            for (int64_t group_id = 0; group_id < group_count; ++group_id) {
                for (int64_t local_id = 0; local_id < group_sizes[group_id];
                     ++local_id, ++global_id) {
                    CUSOLVER_ERROR_FUNC_T(Func_name, func, err, handle,
                                          get_cublas_operation(trans[group_id]), n[group_id],
                                          nrhs[group_id], a_[global_id], lda[group_id],
                                          ipiv32[global_id], b_[global_id], ldb[group_id], nullptr);
                }
            }
            for (int64_t i = 0; i < batch_size; ++i)
                sycl::free(ipiv32[i], queue);
            free(ipiv32);
        });
    });

    return done;
}

// Scratchpad memory not needed as parts of buffer a is used as workspace memory
#define GETRS_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                        \
    sycl::event getrs_batch(                                                                    \
        sycl::queue &queue, oneapi::mkl::transpose *trans, std::int64_t *n, std::int64_t *nrhs, \
        TYPE **a, std::int64_t *lda, std::int64_t **ipiv, TYPE **b, std::int64_t *ldb,          \
        std::int64_t group_count, std::int64_t *group_sizes, TYPE *scratchpad,                  \
        std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {           \
        return getrs_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, trans, n, nrhs, a, lda,  \
                           ipiv, b, ldb, group_count, group_sizes, scratchpad, scratchpad_size, \
                           dependencies);                                                       \
    }

GETRS_BATCH_LAUNCHER_USM(float, cusolverDnSgetrs)
GETRS_BATCH_LAUNCHER_USM(double, cusolverDnDgetrs)
GETRS_BATCH_LAUNCHER_USM(std::complex<float>, cusolverDnCgetrs)
GETRS_BATCH_LAUNCHER_USM(std::complex<double>, cusolverDnZgetrs)

#undef GETRS_BATCH_LAUNCHER_USM

sycl::event orgqr_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        float *a, std::int64_t lda, std::int64_t stride_a, float *tau,
                        std::int64_t stride_tau, std::int64_t batch_size, float *scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "orgqr_batch");
}
sycl::event orgqr_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        double *a, std::int64_t lda, std::int64_t stride_a, double *tau,
                        std::int64_t stride_tau, std::int64_t batch_size, double *scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "orgqr_batch");
}
sycl::event orgqr_batch(sycl::queue &queue, std::int64_t *m, std::int64_t *n, std::int64_t *k,
                        float **a, std::int64_t *lda, float **tau, std::int64_t group_count,
                        std::int64_t *group_sizes, float *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "orgqr_batch");
}
sycl::event orgqr_batch(sycl::queue &queue, std::int64_t *m, std::int64_t *n, std::int64_t *k,
                        double **a, std::int64_t *lda, double **tau, std::int64_t group_count,
                        std::int64_t *group_sizes, double *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "orgqr_batch");
}

template <typename Func, typename T>
inline sycl::event potrf_batch(const char *func_name, Func func, sycl::queue &queue,
                               oneapi::mkl::uplo uplo, std::int64_t n, T *a, std::int64_t lda,
                               std::int64_t stride_a, std::int64_t batch_size, T *scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(n, lda, stride_a, batch_size, scratchpad_size);

    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            int64_t offset = 0;
            cusolverStatus_t err;
            CUresult cuda_result;
            CUdeviceptr a_dev;

            auto *a_ = reinterpret_cast<cuDataType *>(a);

            // Transform ptr and stride to list of ptr's
            cuDataType **a_batched = create_ptr_list_from_stride(a_, stride_a, batch_size);
            CUDA_ERROR_FUNC(cuMemAlloc, cuda_result, &a_dev, sizeof(T *) * batch_size);
            CUDA_ERROR_FUNC(cuMemcpyHtoD, cuda_result, a_dev, a_batched, sizeof(T *) * batch_size);

            auto **a_dev_ = reinterpret_cast<cuDataType **>(a_dev);

            CUSOLVER_ERROR_FUNC_T(func_name, func, err, handle, get_cublas_fill_mode(uplo), (int)n,
                                  a_dev_, (int)lda, nullptr, (int)batch_size);

            cuMemFree(a_dev);
        });
    });
    return done;
}

// Scratchpad memory not needed as parts of buffer a is used as workspace memory
#define POTRF_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                          \
    sycl::event potrf_batch(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, TYPE *a,  \
                            std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size,     \
                            TYPE *scratchpad, std::int64_t scratchpad_size,                       \
                            const std::vector<sycl::event> &dependencies) {                       \
        return potrf_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda, stride_a, \
                           batch_size, scratchpad, scratchpad_size, dependencies);                \
    }

POTRF_BATCH_LAUNCHER_USM(float, cusolverDnSpotrfBatched)
POTRF_BATCH_LAUNCHER_USM(double, cusolverDnDpotrfBatched)
POTRF_BATCH_LAUNCHER_USM(std::complex<float>, cusolverDnCpotrfBatched)
POTRF_BATCH_LAUNCHER_USM(std::complex<double>, cusolverDnZpotrfBatched)

#undef POTRF_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event potrf_batch(const char *func_name, Func func, sycl::queue &queue,
                               oneapi::mkl::uplo *uplo, std::int64_t *n, T **a, std::int64_t *lda,
                               std::int64_t group_count, std::int64_t *group_sizes, T *scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    int64_t batch_size = 0;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(n[i], lda[i], group_sizes[i]);
        batch_size += group_sizes[i];
    }

    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            int64_t offset = 0;
            cusolverStatus_t err;
            CUresult cuda_result;
            CUdeviceptr a_dev;

            CUDA_ERROR_FUNC(cuMemAlloc, cuda_result, &a_dev, sizeof(T *) * batch_size);
            CUDA_ERROR_FUNC(cuMemcpyHtoD, cuda_result, a_dev, a, sizeof(T *) * batch_size);

            auto **a_dev_ = reinterpret_cast<cuDataType **>(a_dev);

            for (int64_t i = 0; i < group_count; i++) {
                CUSOLVER_ERROR_FUNC_T(func_name, func, err, handle, get_cublas_fill_mode(uplo[i]),
                                      (int)n[i], a_dev_ + offset, (int)lda[i], nullptr,
                                      (int)group_sizes[i]);
                offset += group_sizes[i];
            }

            cuMemFree(a_dev);
        });
    });
    return done;
}

// Scratchpad memory not needed as parts of buffer a is used as workspace memory
#define POTRF_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                           \
    sycl::event potrf_batch(                                                                       \
        sycl::queue &queue, oneapi::mkl::uplo *uplo, std::int64_t *n, TYPE **a, std::int64_t *lda, \
        std::int64_t group_count, std::int64_t *group_sizes, TYPE *scratchpad,                     \
        std::int64_t scratchpad_size, const std::vector<sycl::event> &dependencies) {              \
        return potrf_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda,            \
                           group_count, group_sizes, scratchpad, scratchpad_size, dependencies);   \
    }

POTRF_BATCH_LAUNCHER_USM(float, cusolverDnSpotrfBatched)
POTRF_BATCH_LAUNCHER_USM(double, cusolverDnDpotrfBatched)
POTRF_BATCH_LAUNCHER_USM(std::complex<float>, cusolverDnCpotrfBatched)
POTRF_BATCH_LAUNCHER_USM(std::complex<double>, cusolverDnZpotrfBatched)

#undef POTRF_BATCH_LAUNCHER_USM

sycl::event potrs_batch(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, float *a, std::int64_t lda, std::int64_t stride_a,
                        float *b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                        float *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "potrs_batch");
}
sycl::event potrs_batch(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, double *a, std::int64_t lda, std::int64_t stride_a,
                        double *b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                        double *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "potrs_batch");
}
sycl::event potrs_batch(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, std::complex<float> *a, std::int64_t lda,
                        std::int64_t stride_a, std::complex<float> *b, std::int64_t ldb,
                        std::int64_t stride_b, std::int64_t batch_size,
                        std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "potrs_batch");
}
sycl::event potrs_batch(sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, std::complex<double> *a, std::int64_t lda,
                        std::int64_t stride_a, std::complex<double> *b, std::int64_t ldb,
                        std::int64_t stride_b, std::int64_t batch_size,
                        std::complex<double> *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "potrs_batch");
}

template <typename Func, typename T>
inline sycl::event potrs_batch(const char *func_name, Func func, sycl::queue &queue,
                               oneapi::mkl::uplo *uplo, std::int64_t *n, std::int64_t *nrhs, T **a,
                               std::int64_t *lda, T **b, std::int64_t *ldb,
                               std::int64_t group_count, std::int64_t *group_sizes, T *scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    int64_t batch_size = 0;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(n[i], lda[i], group_sizes[i]);
        batch_size += group_sizes[i];

        // cusolver function only supports nrhs = 1
        if (nrhs[i] != 1)
            throw unimplemented("lapack", "potrs_batch",
                                "cusolver potrs_batch only supports nrhs = 1");
    }

    int *info = (int *)malloc_device(sizeof(int *) * batch_size, queue);
    T **a_dev = (T **)malloc_device(sizeof(T *) * batch_size, queue);
    T **b_dev = (T **)malloc_device(sizeof(T *) * batch_size, queue);
    auto done_cpy_a =
        queue.submit([&](sycl::handler &h) { h.memcpy(a_dev, a, batch_size * sizeof(T *)); });

    auto done_cpy_b =
        queue.submit([&](sycl::handler &h) { h.memcpy(b_dev, b, batch_size * sizeof(T *)); });

    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        cgh.depends_on(done_cpy_a);
        cgh.depends_on(done_cpy_b);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            int64_t offset = 0;
            cusolverStatus_t err;
            for (int64_t i = 0; i < group_count; i++) {
                auto **a_ = reinterpret_cast<cuDataType **>(a_dev);
                auto **b_ = reinterpret_cast<cuDataType **>(b_dev);
                auto info_ = reinterpret_cast<int *>(info);
                CUSOLVER_ERROR_FUNC_T(func_name, func, err, handle, get_cublas_fill_mode(uplo[i]),
                                      (int)n[i], (int)nrhs[i], a_ + offset, (int)lda[i],
                                      b_ + offset, (int)ldb[i], info_, (int)group_sizes[i]);
                offset += group_sizes[i];
            }
        });
    });
    return done;
}

// Scratchpad memory not needed as parts of buffer a is used as workspace memory
#define POTRS_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                         \
    sycl::event potrs_batch(                                                                     \
        sycl::queue &queue, oneapi::mkl::uplo *uplo, std::int64_t *n, std::int64_t *nrhs,        \
        TYPE **a, std::int64_t *lda, TYPE **b, std::int64_t *ldb, std::int64_t group_count,      \
        std::int64_t *group_sizes, TYPE *scratchpad, std::int64_t scratchpad_size,               \
        const std::vector<sycl::event> &dependencies) {                                          \
        return potrs_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, nrhs, a, lda, b, \
                           ldb, group_count, group_sizes, scratchpad, scratchpad_size,           \
                           dependencies);                                                        \
    }

POTRS_BATCH_LAUNCHER_USM(float, cusolverDnSpotrsBatched)
POTRS_BATCH_LAUNCHER_USM(double, cusolverDnDpotrsBatched)
POTRS_BATCH_LAUNCHER_USM(std::complex<float>, cusolverDnCpotrsBatched)
POTRS_BATCH_LAUNCHER_USM(std::complex<double>, cusolverDnZpotrsBatched)

#undef POTRS_BATCH_LAUNCHER_USM

sycl::event ungqr_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        std::complex<float> *a, std::int64_t lda, std::int64_t stride_a,
                        std::complex<float> *tau, std::int64_t stride_tau, std::int64_t batch_size,
                        std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "ungqr_batch");
}
sycl::event ungqr_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        std::complex<double> *a, std::int64_t lda, std::int64_t stride_a,
                        std::complex<double> *tau, std::int64_t stride_tau, std::int64_t batch_size,
                        std::complex<double> *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "ungqr_batch");
}
sycl::event ungqr_batch(sycl::queue &queue, std::int64_t *m, std::int64_t *n, std::int64_t *k,
                        std::complex<float> **a, std::int64_t *lda, std::complex<float> **tau,
                        std::int64_t group_count, std::int64_t *group_sizes,
                        std::complex<float> *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "ungqr_batch");
}
sycl::event ungqr_batch(sycl::queue &queue, std::int64_t *m, std::int64_t *n, std::int64_t *k,
                        std::complex<double> **a, std::int64_t *lda, std::complex<double> **tau,
                        std::int64_t group_count, std::int64_t *group_sizes,
                        std::complex<double> *scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event> &dependencies) {
    throw unimplemented("lapack", "ungqr_batch");
}

// BATCH SCRATCHPAD API

template <>
std::int64_t getrf_batch_scratchpad_size<float>(sycl::queue &queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda, std::int64_t stride_a,
                                                std::int64_t stride_ipiv, std::int64_t batch_size) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<double>(sycl::queue &queue, std::int64_t m, std::int64_t n,
                                                 std::int64_t lda, std::int64_t stride_a,
                                                 std::int64_t stride_ipiv,
                                                 std::int64_t batch_size) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<float>>(sycl::queue &queue, std::int64_t m,
                                                              std::int64_t n, std::int64_t lda,
                                                              std::int64_t stride_a,
                                                              std::int64_t stride_ipiv,
                                                              std::int64_t batch_size) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<double>>(sycl::queue &queue, std::int64_t m,
                                                               std::int64_t n, std::int64_t lda,
                                                               std::int64_t stride_a,
                                                               std::int64_t stride_ipiv,
                                                               std::int64_t batch_size) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<float>(sycl::queue &queue, std::int64_t n,
                                                std::int64_t lda, std::int64_t stride_a,
                                                std::int64_t stride_ipiv, std::int64_t batch_size) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<double>(sycl::queue &queue, std::int64_t n,
                                                 std::int64_t lda, std::int64_t stride_a,
                                                 std::int64_t stride_ipiv,
                                                 std::int64_t batch_size) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<float>>(sycl::queue &queue, std::int64_t n,
                                                              std::int64_t lda,
                                                              std::int64_t stride_a,
                                                              std::int64_t stride_ipiv,
                                                              std::int64_t batch_size) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<double>>(sycl::queue &queue, std::int64_t n,
                                                               std::int64_t lda,
                                                               std::int64_t stride_a,
                                                               std::int64_t stride_ipiv,
                                                               std::int64_t batch_size) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}

// cusolverDnXgetrs does not use scratchpad memory
#define GETRS_GROUP_LAUNCHER_SCRATCH(TYPE)                                                    \
    template <>                                                                               \
    std::int64_t getrs_batch_scratchpad_size<TYPE>(                                           \
        sycl::queue & queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs, \
        std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t ldb,  \
        std::int64_t stride_b, std::int64_t batch_size) {                                     \
        return 0;                                                                             \
    }

GETRS_GROUP_LAUNCHER_SCRATCH(float)
GETRS_GROUP_LAUNCHER_SCRATCH(double)
GETRS_GROUP_LAUNCHER_SCRATCH(std::complex<float>)
GETRS_GROUP_LAUNCHER_SCRATCH(std::complex<double>)

#undef GETRS_GROUP_LAUNCHER_SCRATCH

template <>
std::int64_t geqrf_batch_scratchpad_size<float>(sycl::queue &queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda, std::int64_t stride_a,
                                                std::int64_t stride_tau, std::int64_t batch_size) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<double>(sycl::queue &queue, std::int64_t m, std::int64_t n,
                                                 std::int64_t lda, std::int64_t stride_a,
                                                 std::int64_t stride_tau, std::int64_t batch_size) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<float>>(sycl::queue &queue, std::int64_t m,
                                                              std::int64_t n, std::int64_t lda,
                                                              std::int64_t stride_a,
                                                              std::int64_t stride_tau,
                                                              std::int64_t batch_size) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<double>>(sycl::queue &queue, std::int64_t m,
                                                               std::int64_t n, std::int64_t lda,
                                                               std::int64_t stride_a,
                                                               std::int64_t stride_tau,
                                                               std::int64_t batch_size) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}

// cusolverDnXpotrfBatched does not use scratchpad memory
#define POTRF_GROUP_LAUNCHER_SCRATCH(TYPE)                                             \
    template <>                                                                        \
    std::int64_t potrf_batch_scratchpad_size<TYPE>(                                    \
        sycl::queue & queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda, \
        std::int64_t stride_a, std::int64_t batch_size) {                              \
        return 0;                                                                      \
    }

POTRF_GROUP_LAUNCHER_SCRATCH(float)
POTRF_GROUP_LAUNCHER_SCRATCH(double)
POTRF_GROUP_LAUNCHER_SCRATCH(std::complex<float>)
POTRF_GROUP_LAUNCHER_SCRATCH(std::complex<double>)

#undef POTRF_GROUP_LAUNCHER_SCRATCH

template <>
std::int64_t potrs_batch_scratchpad_size<float>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t ldb,
                                                std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("lapack", "potrs_batch_scratchpad_size");
}
template <>
std::int64_t potrs_batch_scratchpad_size<double>(sycl::queue &queue, oneapi::mkl::uplo uplo,
                                                 std::int64_t n, std::int64_t nrhs,
                                                 std::int64_t lda, std::int64_t stride_a,
                                                 std::int64_t ldb, std::int64_t stride_b,
                                                 std::int64_t batch_size) {
    throw unimplemented("lapack", "potrs_batch_scratchpad_size");
}
template <>
std::int64_t potrs_batch_scratchpad_size<std::complex<float>>(
    sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, std::int64_t lda,
    std::int64_t stride_a, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("lapack", "potrs_batch_scratchpad_size");
}
template <>
std::int64_t potrs_batch_scratchpad_size<std::complex<double>>(
    sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, std::int64_t lda,
    std::int64_t stride_a, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("lapack", "potrs_batch_scratchpad_size");
}
template <>
std::int64_t orgqr_batch_scratchpad_size<float>(sycl::queue &queue, std::int64_t m, std::int64_t n,
                                                std::int64_t k, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t stride_tau,
                                                std::int64_t batch_size) {
    throw unimplemented("lapack", "orgqr_batch_scratchpad_size");
}
template <>
std::int64_t orgqr_batch_scratchpad_size<double>(sycl::queue &queue, std::int64_t m, std::int64_t n,
                                                 std::int64_t k, std::int64_t lda,
                                                 std::int64_t stride_a, std::int64_t stride_tau,
                                                 std::int64_t batch_size) {
    throw unimplemented("lapack", "orgqr_batch_scratchpad_size");
}
template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<float>>(
    sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
    std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) {
    throw unimplemented("lapack", "ungqr_batch_scratchpad_size");
}
template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<double>>(
    sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
    std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) {
    throw unimplemented("lapack", "ungqr_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<float>(sycl::queue &queue, std::int64_t *m,
                                                std::int64_t *n, std::int64_t *lda,
                                                std::int64_t group_count,
                                                std::int64_t *group_sizes) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<double>(sycl::queue &queue, std::int64_t *m,
                                                 std::int64_t *n, std::int64_t *lda,
                                                 std::int64_t group_count,
                                                 std::int64_t *group_sizes) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<float>>(sycl::queue &queue, std::int64_t *m,
                                                              std::int64_t *n, std::int64_t *lda,
                                                              std::int64_t group_count,
                                                              std::int64_t *group_sizes) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<double>>(sycl::queue &queue, std::int64_t *m,
                                                               std::int64_t *n, std::int64_t *lda,
                                                               std::int64_t group_count,
                                                               std::int64_t *group_sizes) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<float>(sycl::queue &queue, std::int64_t *n,
                                                std::int64_t *lda, std::int64_t group_count,
                                                std::int64_t *group_sizes) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<double>(sycl::queue &queue, std::int64_t *n,
                                                 std::int64_t *lda, std::int64_t group_count,
                                                 std::int64_t *group_sizes) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<float>>(sycl::queue &queue, std::int64_t *n,
                                                              std::int64_t *lda,
                                                              std::int64_t group_count,
                                                              std::int64_t *group_sizes) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<double>>(sycl::queue &queue, std::int64_t *n,
                                                               std::int64_t *lda,
                                                               std::int64_t group_count,
                                                               std::int64_t *group_sizes) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}

#define GETRS_GROUP_LAUNCHER_SCRATCH(TYPE)                                                     \
    template <>                                                                                \
    std::int64_t getrs_batch_scratchpad_size<TYPE>(                                            \
        sycl::queue & queue, oneapi::mkl::transpose * trans, std::int64_t * n,                 \
        std::int64_t * nrhs, std::int64_t * lda, std::int64_t * ldb, std::int64_t group_count, \
        std::int64_t * group_sizes) {                                                          \
        return 0;                                                                              \
    }

GETRS_GROUP_LAUNCHER_SCRATCH(float)
GETRS_GROUP_LAUNCHER_SCRATCH(double)
GETRS_GROUP_LAUNCHER_SCRATCH(std::complex<float>)
GETRS_GROUP_LAUNCHER_SCRATCH(std::complex<double>)

#undef GETRS_GROUP_LAUNCHER_SCRATCH

template <>
std::int64_t geqrf_batch_scratchpad_size<float>(sycl::queue &queue, std::int64_t *m,
                                                std::int64_t *n, std::int64_t *lda,
                                                std::int64_t group_count,
                                                std::int64_t *group_sizes) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<double>(sycl::queue &queue, std::int64_t *m,
                                                 std::int64_t *n, std::int64_t *lda,
                                                 std::int64_t group_count,
                                                 std::int64_t *group_sizes) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<float>>(sycl::queue &queue, std::int64_t *m,
                                                              std::int64_t *n, std::int64_t *lda,
                                                              std::int64_t group_count,
                                                              std::int64_t *group_sizes) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<double>>(sycl::queue &queue, std::int64_t *m,
                                                               std::int64_t *n, std::int64_t *lda,
                                                               std::int64_t group_count,
                                                               std::int64_t *group_sizes) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t orgqr_batch_scratchpad_size<float>(sycl::queue &queue, std::int64_t *m,
                                                std::int64_t *n, std::int64_t *k, std::int64_t *lda,
                                                std::int64_t group_count,
                                                std::int64_t *group_sizes) {
    throw unimplemented("lapack", "orgqr_batch_scratchpad_size");
}
template <>
std::int64_t orgqr_batch_scratchpad_size<double>(sycl::queue &queue, std::int64_t *m,
                                                 std::int64_t *n, std::int64_t *k,
                                                 std::int64_t *lda, std::int64_t group_count,
                                                 std::int64_t *group_sizes) {
    throw unimplemented("lapack", "orgqr_batch_scratchpad_size");
}

// cusolverDnXpotrfBatched does not use scratchpad memory
#define POTRF_GROUP_LAUNCHER_SCRATCH(TYPE)                                                   \
    template <>                                                                              \
    std::int64_t potrf_batch_scratchpad_size<TYPE>(                                          \
        sycl::queue & queue, oneapi::mkl::uplo * uplo, std::int64_t * n, std::int64_t * lda, \
        std::int64_t group_count, std::int64_t * group_sizes) {                              \
        return 0;                                                                            \
    }

POTRF_GROUP_LAUNCHER_SCRATCH(float)
POTRF_GROUP_LAUNCHER_SCRATCH(double)
POTRF_GROUP_LAUNCHER_SCRATCH(std::complex<float>)
POTRF_GROUP_LAUNCHER_SCRATCH(std::complex<double>)

#undef POTRF_GROUP_LAUNCHER_SCRATCH

// cusolverDnXpotrsBatched does not use scratchpad memory
#define POTRS_GROUP_LAUNCHER_SCRATCH(TYPE)                                                    \
    template <>                                                                               \
    std::int64_t potrs_batch_scratchpad_size<TYPE>(                                           \
        sycl::queue & queue, oneapi::mkl::uplo * uplo, std::int64_t * n, std::int64_t * nrhs, \
        std::int64_t * lda, std::int64_t * ldb, std::int64_t group_count,                     \
        std::int64_t * group_sizes) {                                                         \
        return 0;                                                                             \
    }

POTRS_GROUP_LAUNCHER_SCRATCH(float)
POTRS_GROUP_LAUNCHER_SCRATCH(double)
POTRS_GROUP_LAUNCHER_SCRATCH(std::complex<float>)
POTRS_GROUP_LAUNCHER_SCRATCH(std::complex<double>)

#undef POTRS_GROUP_LAUNCHER_SCRATCH

template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<float>>(sycl::queue &queue, std::int64_t *m,
                                                              std::int64_t *n, std::int64_t *k,
                                                              std::int64_t *lda,
                                                              std::int64_t group_count,
                                                              std::int64_t *group_sizes) {
    throw unimplemented("lapack", "ungqr_batch_scratchpad_size");
}
template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<double>>(sycl::queue &queue, std::int64_t *m,
                                                               std::int64_t *n, std::int64_t *k,
                                                               std::int64_t *lda,
                                                               std::int64_t group_count,
                                                               std::int64_t *group_sizes) {
    throw unimplemented("lapack", "ungqr_batch_scratchpad_size");
}

} // namespace cusolver
} // namespace lapack
} // namespace mkl
} // namespace oneapi
