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
25 66 27182
32 92 27182
76 87 27182
89 89 27182
25 66 27182
32 92 27182
89 89 27182
)";

template <typename mem_T>
bool accuracy(const sycl::device& dev, int64_t n, int64_t lda, uint64_t seed) {
    using fp = typename mem_T_info<mem_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A_initial(lda * n);
    rand_matrix(seed, oneapi::mkl::transpose::nontrans, n, n, A_initial, lda);

    std::vector<fp> A = A_initial;
    std::vector<int64_t> ipiv(n);

    auto info = reference::getrf(n, n, A.data(), lda, ipiv.data());
    if (info != 0) {
        global::log << "Reference getrf failed with info = " << info << std::endl;
        return false;
    }

    /* Compute on device */
    {
        sycl::queue queue{ dev };

        auto A_dev = device_alloc<mem_T>(queue, A.size());
        auto ipiv_dev = device_alloc<mem_T, int64_t>(queue, ipiv.size());
#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::getri_scratchpad_size<fp>(queue, n, lda);
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue, scratchpad_size = oneapi::mkl::lapack::getri_scratchpad_size<fp>,
                           n, lda);
#endif
        auto scratchpad_dev = device_alloc<mem_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, ipiv.data(), ipiv_dev, ipiv.size());
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::mkl::lapack::getri(queue, n, A_dev, lda, ipiv_dev, scratchpad_dev, scratchpad_size);
#else
        TEST_RUN_CT_SELECT(queue, oneapi::mkl::lapack::getri, n, A_dev, lda, ipiv_dev,
                           scratchpad_dev, scratchpad_size);
#endif
        queue.wait_and_throw();

        device_to_host_copy(queue, A_dev, A.data(), A.size());
        queue.wait_and_throw();

        device_free(queue, A_dev);
        device_free(queue, ipiv_dev);
        device_free(queue, scratchpad_dev);
    }

    return check_getri_accuracy(n, A.data(), lda, ipiv.data(), A_initial.data());
}

const char* dependency_input = R"(
1 1 1
)";

template <typename mem_T>
bool usm_dependency(const sycl::device& dev, int64_t n, int64_t lda, uint64_t seed) {
    using fp = typename mem_T_info<mem_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A_initial(lda * n);
    rand_matrix(seed, oneapi::mkl::transpose::nontrans, n, n, A_initial, lda);

    std::vector<fp> A = A_initial;
    std::vector<int64_t> ipiv(n);

    int64_t info = 0;
    info = reference::getrf(n, n, A.data(), lda, ipiv.data());
    if (info != 0) {
        global::log << "Reference getrf failed with info = " << info << std::endl;
        return false;
    }

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{ dev };

        auto A_dev = device_alloc<mem_T>(queue, A.size());
        auto ipiv_dev = device_alloc<mem_T, int64_t>(queue, ipiv.size());
#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::getri_scratchpad_size<fp>(queue, n, lda);
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue, scratchpad_size = oneapi::mkl::lapack::getri_scratchpad_size<fp>,
                           n, lda);
#endif
        auto scratchpad_dev = device_alloc<mem_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, ipiv.data(), ipiv_dev, ipiv.size());
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependent_event(queue);
#ifdef CALL_RT_API
        sycl::event func_event = oneapi::mkl::lapack::getri(
            queue, n, A_dev, lda, ipiv_dev, scratchpad_dev, scratchpad_size,
            sycl::vector_class<sycl::event>{ in_event });
#else
        sycl::event func_event;
        TEST_RUN_CT_SELECT(queue, sycl::event func_event = oneapi::mkl::lapack::getri, n, A_dev,
                           lda, ipiv_dev, scratchpad_dev, scratchpad_size,
                           sycl::vector_class<sycl::event>{ in_event });
#endif
        result = check_dependency(queue, in_event, func_event);

        queue.wait_and_throw();
        device_free(queue, A_dev);
        device_free(queue, ipiv_dev);
        device_free(queue, scratchpad_dev);
    }

    return result;
}

InputTestController<decltype(::accuracy<void>)> accuracy_controller{ accuracy_input };
InputTestController<decltype(::usm_dependency<void>)> dependency_controller{ dependency_input };

} /* unnamed namespace */

#include <gtest/gtest.h>
extern std::vector<sycl::device*> devices;
class GetriTests : public ::testing::TestWithParam<sycl::device*> {};
INSTANTIATE_TEST_SUITE_P(GetriTestSuite, GetriTests, ::testing::ValuesIn(devices),
                         DeviceNamePrint());
RUN_SUITE(Getri)
