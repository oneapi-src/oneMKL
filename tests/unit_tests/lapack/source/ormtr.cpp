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
#include "lapack_reference_wrappers.hpp"
#include "test_helper.hpp"

namespace {

const char* accuracy_input = R"(
1 31 33 35 37 27182
)";

template <typename data_T>
bool accuracy(const sycl::device& dev, oneapi::mkl::uplo uplo, int64_t m, int64_t n, int64_t lda,
              int64_t ldc, uint64_t seed) {
    using fp = typename data_T_info<data_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    oneapi::mkl::side side = oneapi::mkl::side::right;
    oneapi::mkl::transpose trans = oneapi::mkl::transpose::nontrans;

    std::vector<fp> A(n * lda);
    rand_matrix(seed, oneapi::mkl::transpose::nontrans, n, n, A, lda);

    std::vector<fp> tau(n);
    std::vector<fp_real> d(n);
    std::vector<fp_real> e(n);
    auto info = reference::sytrd(uplo, n, A.data(), lda, d.data(), e.data(), tau.data());
    if (0 != info) {
        global::log << "reference sytrd failed with info = " << info << std::endl;
        return false;
    }

    std::vector<fp> C(n * ldc);
    rand_matrix(seed, oneapi::mkl::transpose::nontrans, m, n, C, ldc);
    std::vector<fp> C_initial = C;

    /* Compute on device */
    {
        sycl::queue queue{ dev, async_error_handler };

        auto A_dev = device_alloc<data_T>(queue, A.size());
        auto tau_dev = device_alloc<data_T>(queue, tau.size());
        auto C_dev = device_alloc<data_T>(queue, C.size());
#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::ormtr_scratchpad_size<fp>(
            queue, side, uplo, trans, m, n, lda, ldc);
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue, scratchpad_size = oneapi::mkl::lapack::ormtr_scratchpad_size<fp>,
                           side, uplo, trans, m, n, lda, ldc);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, tau.data(), tau_dev, tau.size());
        host_to_device_copy(queue, C.data(), C_dev, C.size());
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::mkl::lapack::ormtr(queue, side, uplo, trans, m, n, A_dev, lda, tau_dev, C_dev, ldc,
                                   scratchpad_dev, scratchpad_size);
#else
        TEST_RUN_CT_SELECT(queue, oneapi::mkl::lapack::ormtr, side, uplo, trans, m, n, A_dev, lda,
                           tau_dev, C_dev, ldc, scratchpad_dev, scratchpad_size);
#endif
        queue.wait_and_throw();

        device_to_host_copy(queue, A_dev, A.data(), A.size());
        device_to_host_copy(queue, tau_dev, tau.data(), tau.size());
        device_to_host_copy(queue, C_dev, C.data(), C.size());
        queue.wait_and_throw();

        device_free(queue, A_dev);
        device_free(queue, tau_dev);
        device_free(queue, C_dev);
    }
    bool result = true;

    auto& C_ref = C_initial;
    info = reference::ormtr(side, uplo, trans, m, n, A.data(), lda, tau.data(), C_ref.data(), ldc);
    if (0 != info) {
        global::log << "reference ormtr failed with info = " << info << std::endl;
        return false;
    }
    if (!rel_mat_err_check(m, n, C.data(), ldc, C_ref.data(), ldc)) {
        global::log << "Multiplication check failed" << std::endl;
        result = false;
    }

    return result;
}

const char* dependency_input = R"(
1 1 1 1 1 1
)";

template <typename data_T>
bool usm_dependency(const sycl::device& dev, oneapi::mkl::uplo uplo, int64_t m, int64_t n,
                    int64_t lda, int64_t ldc, uint64_t seed) {
    using fp = typename data_T_info<data_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    oneapi::mkl::side side = oneapi::mkl::side::right;
    oneapi::mkl::transpose trans = oneapi::mkl::transpose::nontrans;

    std::vector<fp> A(n * lda);
    rand_matrix(seed, oneapi::mkl::transpose::nontrans, n, n, A, lda);

    std::vector<fp> tau(n);
    std::vector<fp_real> d(n);
    std::vector<fp_real> e(n);
    auto info = reference::sytrd(uplo, n, A.data(), lda, d.data(), e.data(), tau.data());
    if (0 != info) {
        global::log << "reference sytrd failed with info = " << info << std::endl;
        return false;
    }

    std::vector<fp> C(n * ldc);
    rand_matrix(seed, oneapi::mkl::transpose::nontrans, m, n, C, ldc);
    std::vector<fp> C_initial = C;

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{ dev, async_error_handler };

        auto A_dev = device_alloc<data_T>(queue, A.size());
        auto tau_dev = device_alloc<data_T>(queue, tau.size());
        auto C_dev = device_alloc<data_T>(queue, C.size());
#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::ormtr_scratchpad_size<fp>(
            queue, side, uplo, trans, m, n, lda, ldc);
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue, scratchpad_size = oneapi::mkl::lapack::ormtr_scratchpad_size<fp>,
                           side, uplo, trans, m, n, lda, ldc);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, tau.data(), tau_dev, tau.size());
        host_to_device_copy(queue, C.data(), C_dev, C.size());
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependent_event(queue);
#ifdef CALL_RT_API
        sycl::event func_event = oneapi::mkl::lapack::ormtr(
            queue, side, uplo, trans, m, n, A_dev, lda, tau_dev, C_dev, ldc, scratchpad_dev,
            scratchpad_size, sycl::vector_class<sycl::event>{ in_event });
#else
        sycl::event func_event;
        TEST_RUN_CT_SELECT(queue, sycl::event func_event = oneapi::mkl::lapack::ormtr, side, uplo,
                           trans, m, n, A_dev, lda, tau_dev, C_dev, ldc, scratchpad_dev,
                           scratchpad_size, sycl::vector_class<sycl::event>{ in_event });
#endif
        result = check_dependency(queue, in_event, func_event);
        queue.wait_and_throw();

        queue.wait_and_throw();
        device_free(queue, A_dev);
        device_free(queue, tau_dev);
        device_free(queue, C_dev);
    }

    return result;
}

InputTestController<decltype(::accuracy<void>)> accuracy_controller{ accuracy_input };
InputTestController<decltype(::usm_dependency<void>)> dependency_controller{ dependency_input };

} /* unnamed namespace */

#include <gtest/gtest.h>
extern std::vector<sycl::device*> devices;
class OrmtrTests : public ::testing::TestWithParam<sycl::device*> {};
INSTANTIATE_TEST_SUITE_P(OrmtrTestSuite, OrmtrTests, ::testing::ValuesIn(devices),
                         DeviceNamePrint());
RUN_SUITE_REAL(Ormtr)
