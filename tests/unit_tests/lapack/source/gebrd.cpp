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

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl.hpp"
#include "lapack_common.hpp"
#include "lapack_test_controller.hpp"
#include "lapack_accuracy_checks.hpp"
#include "lapack_reference_wrappers.hpp"
#include "test_helper.hpp"

namespace {

const char* accuracy_input = R"(
29 37 34 27182
27 25 33 27182
)";

template <typename data_T>
bool accuracy(const sycl::device& dev, int64_t m, int64_t n, int64_t lda, uint64_t seed) {
    using fp = typename data_T_info<data_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    int64_t min_mn = std::min<int64_t>(m, n);

    std::vector<fp> A_initial(lda * n);
    rand_matrix(seed, oneapi::mkl::transpose::nontrans, m, n, A_initial, lda);

    std::vector<fp> A = A_initial;
    std::vector<fp_real> d(min_mn);
    std::vector<fp_real> e(std::max<int64_t>(min_mn - 1, 1));

    std::vector<fp> tauq(min_mn);
    std::vector<fp> taup(min_mn);

    /* Compute on device */
    {
        sycl::queue queue{ dev, async_error_handler };

        auto A_dev = device_alloc<data_T>(queue, A.size());
        auto d_dev = device_alloc<data_T, fp_real>(queue, d.size());
        auto e_dev = device_alloc<data_T, fp_real>(queue, e.size());
        auto tauq_dev = device_alloc<data_T>(queue, tauq.size());
        auto taup_dev = device_alloc<data_T>(queue, taup.size());

#ifdef CALL_RT_API
        const auto scratchpad_size =
            oneapi::mkl::lapack::gebrd_scratchpad_size<fp>(queue, m, n, lda);
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(
            queue, scratchpad_size = oneapi::mkl::lapack::gebrd_scratchpad_size<fp>, m, n, lda);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::mkl::lapack::gebrd(queue, m, n, A_dev, lda, d_dev, e_dev, tauq_dev, taup_dev,
                                   scratchpad_dev, scratchpad_size);
#else
        TEST_RUN_LAPACK_CT_SELECT(queue, oneapi::mkl::lapack::gebrd, m, n, A_dev, lda, d_dev, e_dev,
                                  tauq_dev, taup_dev, scratchpad_dev, scratchpad_size);
#endif
        queue.wait_and_throw();

        device_to_host_copy(queue, A_dev, A.data(), A.size());
        device_to_host_copy(queue, d_dev, d.data(), d.size());
        device_to_host_copy(queue, e_dev, e.data(), e.size());
        device_to_host_copy(queue, tauq_dev, tauq.data(), tauq.size());
        device_to_host_copy(queue, taup_dev, taup.data(), taup.size());
        queue.wait_and_throw();

        device_free(queue, A_dev);
        device_free(queue, d_dev);
        device_free(queue, e_dev);
        device_free(queue, tauq_dev);
        device_free(queue, taup_dev);
        device_free(queue, scratchpad_dev);
    }

    reference::gebrd(m, n, A_initial.data(), lda, d.data(), e.data(), tauq.data(), taup.data());
    return rel_mat_err_check<fp>(m, n, A, lda, A_initial, lda, 30.0);
}

const char* dependency_input = R"(
1 1 1 1
)";

template <typename data_T>
bool usm_dependency(const sycl::device& dev, int64_t m, int64_t n, int64_t lda, uint64_t seed) {
    using fp = typename data_T_info<data_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    int64_t min_mn = std::min<int64_t>(m, n);

    std::vector<fp> A_initial(lda * n);
    rand_matrix(seed, oneapi::mkl::transpose::nontrans, m, n, A_initial, lda);

    auto A = A_initial;
    std::vector<fp_real> d(min_mn);
    std::vector<fp_real> e(std::max<int64_t>(min_mn - 1, 1));
    std::vector<fp> tauq(min_mn);
    std::vector<fp> taup(min_mn);

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{ dev, async_error_handler };

        auto A_dev = device_alloc<data_T>(queue, A.size());
        auto d_dev = device_alloc<data_T, fp_real>(queue, d.size());
        auto e_dev = device_alloc<data_T, fp_real>(queue, e.size());
        auto tauq_dev = device_alloc<data_T>(queue, tauq.size());
        auto taup_dev = device_alloc<data_T>(queue, taup.size());

#ifdef CALL_RT_API
        const auto scratchpad_size =
            oneapi::mkl::lapack::gebrd_scratchpad_size<fp>(queue, m, n, lda);
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(
            queue, scratchpad_size = oneapi::mkl::lapack::gebrd_scratchpad_size<fp>, m, n, lda);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependency(queue);
#ifdef CALL_RT_API
        sycl::event func_event = oneapi::mkl::lapack::gebrd(
            queue, m, n, A_dev, lda, d_dev, e_dev, tauq_dev, taup_dev, scratchpad_dev,
            scratchpad_size, std::vector<sycl::event>{ in_event });
#else
        sycl::event func_event;
        TEST_RUN_LAPACK_CT_SELECT(queue, func_event = oneapi::mkl::lapack::gebrd, m, n, A_dev, lda,
                                  d_dev, e_dev, tauq_dev, taup_dev, scratchpad_dev, scratchpad_size,
                                  std::vector<sycl::event>{ in_event });
#endif
        result = check_dependency(queue, in_event, func_event);

        queue.wait_and_throw();
        device_free(queue, A_dev);
        device_free(queue, d_dev);
        device_free(queue, e_dev);
        device_free(queue, tauq_dev);
        device_free(queue, taup_dev);
        device_free(queue, scratchpad_dev);
    }

    return result;
}

InputTestController<decltype(::accuracy<void>)> accuracy_controller{ accuracy_input };
InputTestController<decltype(::usm_dependency<void>)> dependency_controller{ dependency_input };

} /* anonymous namespace */

#include "lapack_gtest_suite.hpp"
INSTANTIATE_GTEST_SUITE_ACCURACY(Gebrd);
INSTANTIATE_GTEST_SUITE_DEPENDENCY(Gebrd);
