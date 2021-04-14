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

namespace {

const char* accuracy_input = R"(
1 0 27 33 27182
)";

template <typename mem_T>
bool accuracy(const sycl::device& dev, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, int64_t n,
              int64_t lda, uint64_t seed) {
    using fp = typename mem_T_info<mem_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A(lda * n);
    rand_hermitian_matrix(seed, uplo, n, A, lda);

    std::vector<fp> A_initial = A;
    std::vector<fp_real> w(n);

    /* Compute on device */
    {
        sycl::queue queue{ dev };
        auto A_dev = device_alloc<mem_T>(queue, A.size());
        auto w_dev = device_alloc<mem_T, fp_real>(queue, w.size());
#ifdef CALL_RT_API
        const auto scratchpad_size =
            oneapi::mkl::lapack::heevd_scratchpad_size<fp>(queue, jobz, uplo, n, lda);
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue, scratchpad_size = oneapi::mkl::lapack::heevd_scratchpad_size<fp>,
                           jobz, uplo, n, lda);
#endif
        auto scratchpad_dev = device_alloc<mem_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::mkl::lapack::heevd(queue, jobz, uplo, n, A_dev, lda, w_dev, scratchpad_dev,
                                   scratchpad_size);
#else
        TEST_RUN_CT_SELECT(queue, oneapi::mkl::lapack::heevd, jobz, uplo, n, A_dev, lda, w_dev,
                           scratchpad_dev, scratchpad_size);
#endif
        queue.wait_and_throw();

        device_to_host_copy(queue, A_dev, A.data(), A.size());
        device_to_host_copy(queue, w_dev, w.data(), w.size());
        queue.wait_and_throw();

        device_free(queue, A_dev);
        device_free(queue, w_dev);
        device_free(queue, scratchpad_dev);
    }

    const auto& Z = A;
    auto ldz = lda;
    const auto& D = w;
    hermitian_to_full(uplo, n, A_initial, lda);
    bool result = true;

    /* |D_ref - D| < |D_ref| O(eps) */
    std::vector<fp_real> D_ref(n);
    reference::heevd(oneapi::mkl::job::novec, uplo, n, std::vector<fp>(A_initial).data(), lda,
                     D_ref.data());
    if (!rel_vec_err_check(n, D_ref.data(), D.data(), 10.0)) {
        global::log << "Eigenvalue check failed" << std::endl;
        result = false;
    }

    if (oneapi::mkl::job::vec == jobz) {
        /* |A - Z D Z'| < |A| O(eps) */
        std::vector<fp> ZD(n * n);
        int64_t ldzd = n;
        std::vector<fp> ZDZ(n * n);
        int64_t ldzdz = n;
        for (int64_t col = 0; col < n; col++)
            for (int64_t row = 0; row < n; row++)
                ZD[row + col * ldzd] = Z[row + col * ldz] * D[col];
        reference::gemm(oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::conjtrans, n, n,
                        n, 1.0, ZD.data(), ldzd, Z.data(), ldz, 0.0, ZDZ.data(), ldzdz);

        if (!rel_mat_err_check(n, n, A_initial.data(), lda, ZDZ.data(), ldzdz)) {
            global::log << "Factorization check failed" << std::endl;
            result = false;
        }

        /* |I - Z Z'| < n O(eps) */
        std::vector<fp> ZZ(n * n);
        int64_t ldzz = n;
        reference::sy_he_rk(oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, n, n, 1.0,
                            Z.data(), ldz, 0.0, ZZ.data(), ldzz);
        hermitian_to_full(oneapi::mkl::uplo::upper, n, ZZ, ldzz);
        if (!rel_id_err_check(n, ZZ.data(), ldzz)) {
            global::log << "Orthogonality check failed" << std::endl;
            result = false;
        }
    }
    return result;
}

const char* dependency_input = R"(
1 1 1 1 1
)";

template <typename mem_T>
bool usm_dependency(const sycl::device& dev, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                    int64_t n, int64_t lda, uint64_t seed) {
    using fp = typename mem_T_info<mem_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A(lda * n);
    rand_hermitian_matrix(seed, uplo, n, A, lda);

    std::vector<fp> A_initial = A;
    std::vector<fp_real> w(n);

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{ dev };
        auto A_dev = device_alloc<mem_T>(queue, A.size());
        auto w_dev = device_alloc<mem_T, fp_real>(queue, w.size());
#ifdef CALL_RT_API
        const auto scratchpad_size =
            oneapi::mkl::lapack::heevd_scratchpad_size<fp>(queue, jobz, uplo, n, lda);
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue, scratchpad_size = oneapi::mkl::lapack::heevd_scratchpad_size<fp>,
                           jobz, uplo, n, lda);
#endif
        auto scratchpad_dev = device_alloc<mem_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependent_event(queue);
#ifdef CALL_RT_API
        sycl::event func_event = oneapi::mkl::lapack::heevd(
            queue, jobz, uplo, n, A_dev, lda, w_dev, scratchpad_dev, scratchpad_size,
            sycl::vector_class<sycl::event>{ in_event });
#else
        sycl::event func_event;
        TEST_RUN_CT_SELECT(queue, sycl::event func_event = oneapi::mkl::lapack::heevd, jobz, uplo,
                           n, A_dev, lda, w_dev, scratchpad_dev, scratchpad_size,
                           sycl::vector_class<sycl::event>{ in_event });
#endif
        result = check_dependency(queue, in_event, func_event);

        queue.wait_and_throw();
        device_free(queue, A_dev);
        device_free(queue, w_dev);
        device_free(queue, scratchpad_dev);
    }

    return result;
}

InputTestController<decltype(::accuracy<void>)> accuracy_controller{ accuracy_input };
InputTestController<decltype(::usm_dependency<void>)> dependency_controller{ dependency_input };

} /* unnamed namespace */

#include <gtest/gtest.h>
extern std::vector<sycl::device*> devices;
class HeevdTests : public ::testing::TestWithParam<sycl::device*> {};
INSTANTIATE_TEST_SUITE_P(HeevdTestSuite, HeevdTests, ::testing::ValuesIn(devices),
                         DeviceNamePrint());
RUN_SUITE_COMPLEX(Heevd)
