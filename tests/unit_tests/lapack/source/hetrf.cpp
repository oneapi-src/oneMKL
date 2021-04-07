/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <complex>
#include <vector>

#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"
#include "lapack_common.hpp"
#include "lapack_test_controller.hpp"
#include "lapack_accuracy_checks.hpp"
#include "reference_lapack_wrappers.hpp"
#include "test_helper.hpp"

static const char* accuracy_input = R"(
0 27 33 27182
0 42 45 27182
1 27 33 27182
1 42 45 27182
)";

template <typename mem_T>
bool accuracy(const sycl::device &dev, oneapi::mkl::uplo uplo, int64_t n, int64_t lda, uint64_t seed) {
    using fp = typename mem_T_info<mem_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A(lda*n);
    std::vector<int64_t> ipiv(n);
    rand_hermitian_matrix(seed, uplo, n, A, lda);

    std::vector<fp> A_initial = A;

    /* Compute on device */
    {
        sycl::queue queue{dev};

        auto A_dev    = device_alloc<mem_T>(queue, A.size());
        auto ipiv_dev = device_alloc<mem_T, int64_t>(queue, ipiv.size());
#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::hetrf_scratchpad_size<fp>(queue, uplo, n, lda);
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue, scratchpad_size = oneapi::mkl::lapack::hetrf_scratchpad_size<fp>, uplo, n, lda);
#endif
        auto scratchpad_dev = device_alloc<mem_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, ipiv.data(), ipiv_dev, ipiv.size());
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::mkl::lapack::hetrf(queue, uplo, n, A_dev, lda, ipiv_dev, scratchpad_dev, scratchpad_size);
#else
        TEST_RUN_CT_SELECT(queue, oneapi::mkl::lapack::hetrf, uplo, n, A_dev, lda, ipiv_dev, scratchpad_dev, scratchpad_size);
#endif
        queue.wait_and_throw();

        device_to_host_copy(queue, A_dev, A.data(), A.size());
        device_to_host_copy(queue, ipiv_dev, ipiv.data(), ipiv.size());
        queue.wait_and_throw();

        device_free(queue, A_dev);
        device_free(queue, ipiv_dev);
        device_free(queue, scratchpad_dev);
    }

    std::vector<fp> U(n*n);
    std::vector<fp> Uk(n*n);
    int64_t ldu = n;
    std::vector<fp> D(n*n);
    int64_t ldd = n;
    hermitian_to_full(uplo, n, A_initial, lda);
    bool result = true;

    for (int64_t d = 0; d < n; d++)
        U[d + d*ldu] = 1.0;

    if (uplo == oneapi::mkl::uplo::upper) {

        int64_t k = n-1;
        while (k >= 0) {
            reference::laset('A', n, n, 0.0, 1.0, Uk.data(), ldu);
            if (ipiv[k] > 0){ /* 1x1 block case */

                auto piv = ipiv[k]-1;
                for(int64_t i = 0; i < k; i++)
                    Uk[i + k*ldu] = A[i + k*lda];
                if (piv != k)
                    reference::swap(n, Uk.data()+(k + 0*ldu), ldu, Uk.data()+(piv + 0*ldu), ldu);
                auto U_temp = U;
                reference::gemm(oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, n, n, n, 1.0, U_temp.data(), ldu, Uk.data(), ldu, 0.0, U.data(), ldu);

                D[k + k*ldd] = A[k + k*lda];
                k -= 1;

            } else { /* 2x2 block case */

                auto piv = -ipiv[k]-1;
                for(int64_t i = 0; i < k-1; i++) {
                    Uk[i + k*ldu] = A[i + k*lda];
                    Uk[i + (k-1)*ldu] = A[i + (k-1)*lda];
                }
                if (piv != k-1)
                    reference::swap(n, Uk.data()+(k-1 + 0*ldu), ldu, Uk.data()+(piv + 0*ldu), ldu);
                auto U_temp = U;
                reference::gemm(oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, n, n, n, 1.0, U_temp.data(), ldu, Uk.data(), ldu, 0.0, U.data(), ldu);

                D[k   +     k*ldd] = A[k   +     k*lda];
                D[k-1 + (k-1)*ldd] = A[k-1 + (k-1)*lda];
                D[k-1 +     k*ldd] = A[k-1 +     k*lda];
                D[k   + (k-1)*ldd] = get_conj(A[k-1 + k*lda]);
                k -= 2;
            }
        }
    } else {
        int64_t k = 0;
        while (k < n) {
            reference::laset('A', n, n, 0.0, 1.0, Uk.data(), ldu);
            if (ipiv[k] > 0){ /* 1x1 block case */

                auto piv = ipiv[k]-1;
                for(int64_t i = k+1; i < n; i++)
                    Uk[i + k*ldu] = A[i + k*lda];
                if (piv != k)
                    reference::swap(n, Uk.data()+(k + 0*lda), ldu, Uk.data()+(piv + 0*ldu), ldu);
                auto U_temp = U;
                reference::gemm(oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, n, n, n, 1.0, U_temp.data(), ldu, Uk.data(), ldu, 0.0, U.data(), ldu);

                D[k + (k)*ldd] = A[k + (k)*lda];
                k += 1;

            } else { /* 2x2 block case */

                auto piv = -ipiv[k]-1;
                for(int64_t i = k+2; i < n; i++) {
                    Uk[i + k*ldu] = A[i + k*lda];
                    Uk[i + (k+1)*ldu] = A[i + (k+1)*lda];
                }
                if (piv != k)
                    reference::swap(n, Uk.data()+(k+1 + 0*ldu), ldu, Uk.data()+(piv + 0*ldu), ldu);
                auto U_temp = U;
                reference::gemm(oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, n, n, n, 1.0, U_temp.data(), ldu, Uk.data(), ldu, 0.0, U.data(), ldu);

                D[k   +     k*ldd] = A[k   +     k*lda];
                D[k+1 + (k+1)*ldd] = A[k+1 + (k+1)*lda];
                D[k+1 +     k*ldd] = A[k+1 +     k*lda];
                D[k   + (k+1)*ldd] = get_conj(A[k+1 + k*lda]);
                k += 2;
            }
        }
    }


    /* |A - UDU'| < |A| O(eps) */
    std::vector<fp> UD(n*n);
    int64_t ldud = n;
    reference::gemm(oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, n, n, n, 1.0, U.data(), ldu, D.data(), ldd, 0.0, UD.data(), ldud);

    std::vector<fp> UDU(n*n);
    int64_t ldudu = n;
    reference::gemm(oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::conjtrans, n, n, n, 1.0, UD.data(), ldud, U.data(), ldu, 0.0, UDU.data(), ldudu);

    if(!rel_mat_err_check(n, n, UDU.data(), ldudu, A_initial.data(), lda)) {
        global::log << "\tFactorization check failed" << std::endl;
        result = false;
    }

    return result;
}

static const char* dependency_input = R"(
1 1 1 1
)";

template <typename mem_T>
bool usm_dependency(const sycl::device &dev, oneapi::mkl::uplo uplo, int64_t n, int64_t lda, uint64_t seed) {
    using fp = typename mem_T_info<mem_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A(lda*n);
    std::vector<int64_t> ipiv(n);
    rand_hermitian_matrix(seed, uplo, n, A, lda);

    std::vector<fp> A_initial = A;

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{dev};

        auto A_dev    = device_alloc<mem_T>(queue, A.size());
        auto ipiv_dev = device_alloc<mem_T, int64_t>(queue, ipiv.size());
#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::hetrf_scratchpad_size<fp>(queue, uplo, n, lda);
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue, scratchpad_size = oneapi::mkl::lapack::hetrf_scratchpad_size<fp>, uplo, n, lda);
#endif
        auto scratchpad_dev = device_alloc<mem_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, ipiv.data(), ipiv_dev, ipiv.size());
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependent_event(queue);
#ifdef CALL_RT_API
        sycl::event func_event = oneapi::mkl::lapack::hetrf(queue, uplo, n, A_dev, lda, ipiv_dev, scratchpad_dev, scratchpad_size, sycl::vector_class<sycl::event>{in_event});
#else
        sycl::event func_event;
        TEST_RUN_CT_SELECT(queue, sycl::event func_event = oneapi::mkl::lapack::hetrf, uplo, n, A_dev, lda, ipiv_dev, scratchpad_dev, scratchpad_size, sycl::vector_class<sycl::event>{in_event});
#endif
        result = check_dependency(in_event, func_event);

        queue.wait_and_throw();
        device_free(queue, A_dev);
        device_free(queue, ipiv_dev);
        device_free(queue, scratchpad_dev);
    }

    return result;
}

static InputTestController<decltype(::accuracy<void>)> accuracy_controller{accuracy_input};
static InputTestController<decltype(::usm_dependency<void>)> dependency_controller{dependency_input};

#ifdef STANDALONE
int main() {
    sycl::device dev = sycl::device { sycl::host_selector{} };
    int64_t res = 0;
    res += !accuracy_controller.run(::accuracy<ComplexSinglePrecisionUsm>, dev);
    res += !accuracy_controller.run(::accuracy<ComplexDoublePrecisionUsm>, dev);
    res += !accuracy_controller.run(::accuracy<ComplexSinglePrecisionBuffer>, dev);
    res += !accuracy_controller.run(::accuracy<ComplexDoublePrecisionBuffer>, dev);
    res += !dependency_controller.run(::usm_dependency<ComplexSinglePrecisionUsm>, dev);
    res += !dependency_controller.run(::usm_dependency<ComplexDoublePrecisionUsm>, dev);
    return res;
}
#else
#include <gtest/gtest.h>
extern std::vector<sycl::device*> devices;
class HetrfTests : public ::testing::TestWithParam<sycl::device*> {};
INSTANTIATE_TEST_SUITE_P(HetrfTestSuite, HetrfTests, ::testing::ValuesIn(devices), DeviceNamePrint());
RUN_SUITE_COMPLEX(Hetrf)
#endif
