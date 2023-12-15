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
0 30 4 42 31 27182
)";

template <typename data_T>
bool accuracy(const sycl::device& dev, oneapi::mkl::uplo uplo, int64_t n, int64_t nrhs, int64_t lda,
              int64_t ldb, uint64_t seed) {
    using fp = typename data_T_info<data_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A_initial(lda * n);
    std::vector<fp> B_initial(ldb * nrhs);
    rand_pos_def_matrix(seed, uplo, n, A_initial, lda);
    rand_matrix(seed, oneapi::mkl::transpose::nontrans, n, nrhs, B_initial, ldb);

    std::vector<fp> A = A_initial;
    std::vector<fp> B = B_initial;

    auto info = reference::potrf(uplo, n, A.data(), lda);
    if (0 != info) {
        test_log::lout << "reference potrf failed with info = " << info << std::endl;
        return false;
    }

    /* Compute on device */
    {
        sycl::queue queue{ dev, async_error_handler };

        auto A_dev = device_alloc<data_T>(queue, A.size());
        auto B_dev = device_alloc<data_T>(queue, B.size());
#ifdef CALL_RT_API
        const auto scratchpad_size =
            oneapi::mkl::lapack::potrs_scratchpad_size<fp>(queue, uplo, n, nrhs, lda, ldb);
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(queue,
                                  scratchpad_size = oneapi::mkl::lapack::potrs_scratchpad_size<fp>,
                                  uplo, n, nrhs, lda, ldb);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, B.data(), B_dev, B.size());
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::mkl::lapack::potrs(queue, uplo, n, nrhs, A_dev, lda, B_dev, ldb, scratchpad_dev,
                                   scratchpad_size);
#else
        TEST_RUN_LAPACK_CT_SELECT(queue, oneapi::mkl::lapack::potrs, uplo, n, nrhs, A_dev, lda,
                                  B_dev, ldb, scratchpad_dev, scratchpad_size);
#endif
        queue.wait_and_throw();

        device_to_host_copy(queue, B_dev, B.data(), B.size());
        queue.wait_and_throw();

        device_free(queue, A_dev);
        device_free(queue, B_dev);
        device_free(queue, scratchpad_dev);
    }

    return check_potrs_accuracy(uplo, n, nrhs, B, ldb, A_initial, lda, B_initial);
}

const char* dependency_input = R"(
1 1 1 1 1 1
)";

template <typename data_T>
bool usm_dependency(const sycl::device& dev, oneapi::mkl::uplo uplo, int64_t n, int64_t nrhs,
                    int64_t lda, int64_t ldb, uint64_t seed) {
    using fp = typename data_T_info<data_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A_initial(lda * n);
    std::vector<fp> B_initial(ldb * nrhs);
    rand_pos_def_matrix(seed, uplo, n, A_initial, lda);
    rand_matrix(seed, oneapi::mkl::transpose::nontrans, n, nrhs, B_initial, ldb);

    std::vector<fp> A = A_initial;
    std::vector<fp> B = B_initial;

    auto info = reference::potrf(uplo, n, A.data(), lda);
    if (0 != info) {
        test_log::lout << "reference potrf failed with info = " << info << std::endl;
        return false;
    }

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{ dev, async_error_handler };

        auto A_dev = device_alloc<data_T>(queue, A.size());
        auto B_dev = device_alloc<data_T>(queue, B.size());
#ifdef CALL_RT_API
        const auto scratchpad_size =
            oneapi::mkl::lapack::potrs_scratchpad_size<fp>(queue, uplo, n, nrhs, lda, ldb);
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(queue,
                                  scratchpad_size = oneapi::mkl::lapack::potrs_scratchpad_size<fp>,
                                  uplo, n, nrhs, lda, ldb);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, B.data(), B_dev, B.size());
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependency(queue);
#ifdef CALL_RT_API
        sycl::event func_event =
            oneapi::mkl::lapack::potrs(queue, uplo, n, nrhs, A_dev, lda, B_dev, ldb, scratchpad_dev,
                                       scratchpad_size, std::vector<sycl::event>{ in_event });
#else
        sycl::event func_event;
        TEST_RUN_LAPACK_CT_SELECT(queue, func_event = oneapi::mkl::lapack::potrs, uplo, n, nrhs,
                                  A_dev, lda, B_dev, ldb, scratchpad_dev, scratchpad_size,
                                  std::vector<sycl::event>{ in_event });
#endif
        result = check_dependency(queue, in_event, func_event);

        queue.wait_and_throw();
        device_free(queue, A_dev);
        device_free(queue, B_dev);
        device_free(queue, scratchpad_dev);
    }

    return result;
}

InputTestController<decltype(::accuracy<void>)> accuracy_controller{ accuracy_input };
InputTestController<decltype(::usm_dependency<void>)> dependency_controller{ dependency_input };

} /* anonymous namespace */

#include "lapack_gtest_suite.hpp"
INSTANTIATE_GTEST_SUITE_ACCURACY(Potrs);
INSTANTIATE_GTEST_SUITE_DEPENDENCY(Potrs);
