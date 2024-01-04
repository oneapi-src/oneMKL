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
1 4 15 123 3 27182
)";

template <typename data_T>
bool accuracy(const sycl::device& dev, oneapi::mkl::uplo uplo, int64_t n, int64_t lda,
              int64_t stride_a, int64_t batch_size, uint64_t seed) {
    using fp = typename data_T_info<data_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A_initial(stride_a * batch_size);
    for (int64_t i = 0; i < batch_size; i++)
        rand_pos_def_matrix(seed, uplo, n, A_initial, lda, i * stride_a);

    std::vector<fp> A = A_initial;

    /* Compute on device */
    {
        sycl::queue queue{ dev, async_error_handler };

        auto A_dev = device_alloc<data_T>(queue, A.size());
#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::potrf_batch_scratchpad_size<fp>(
            queue, uplo, n, lda, stride_a, batch_size);
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(
            queue, scratchpad_size = oneapi::mkl::lapack::potrf_batch_scratchpad_size<fp>, uplo, n,
            lda, stride_a, batch_size);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::mkl::lapack::potrf_batch(queue, uplo, n, A_dev, lda, stride_a, batch_size,
                                         scratchpad_dev, scratchpad_size);
#else
        TEST_RUN_LAPACK_CT_SELECT(queue, oneapi::mkl::lapack::potrf_batch, uplo, n, A_dev, lda,
                                  stride_a, batch_size, scratchpad_dev, scratchpad_size);
#endif
        queue.wait_and_throw();

        device_to_host_copy(queue, A_dev, A.data(), A.size());
        queue.wait_and_throw();

        device_free(queue, A_dev);
        device_free(queue, scratchpad_dev);
    }

    bool result = true;
    for (int64_t i = 0; i < batch_size; i++) {
        auto A_ = copy_vector(A, lda * n, i * stride_a);
        auto A_initial_ = copy_vector(A_initial, lda * n, i * stride_a);
        if (!check_potrf_accuracy(A_initial_, A_, uplo, n, lda)) {
            test_log::lout << "batch routine index " << i << " failed" << std::endl;
            result = false;
        }
    }

    return result;
}

const char* dependency_input = R"(
1 1 1 1 1 1
)";

template <typename data_T>
bool usm_dependency(const sycl::device& dev, oneapi::mkl::uplo uplo, int64_t n, int64_t lda,
                    int64_t stride_a, int64_t batch_size, uint64_t seed) {
    using fp = typename data_T_info<data_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A_initial(stride_a * batch_size);
    for (int64_t i = 0; i < batch_size; i++)
        rand_pos_def_matrix(seed, uplo, n, A_initial, lda, i * stride_a);

    std::vector<fp> A = A_initial;

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{ dev, async_error_handler };

        auto A_dev = device_alloc<data_T>(queue, A.size());
#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::potrf_batch_scratchpad_size<fp>(
            queue, uplo, n, lda, stride_a, batch_size);
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(
            queue, scratchpad_size = oneapi::mkl::lapack::potrf_batch_scratchpad_size<fp>, uplo, n,
            lda, stride_a, batch_size);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependency(queue);
#ifdef CALL_RT_API
        sycl::event func_event = oneapi::mkl::lapack::potrf_batch(
            queue, uplo, n, A_dev, lda, stride_a, batch_size, scratchpad_dev, scratchpad_size,
            std::vector<sycl::event>{ in_event });
#else
        sycl::event func_event;
        TEST_RUN_LAPACK_CT_SELECT(queue, func_event = oneapi::mkl::lapack::potrf_batch, uplo, n,
                                  A_dev, lda, stride_a, batch_size, scratchpad_dev, scratchpad_size,
                                  std::vector<sycl::event>{ in_event });
#endif
        result = check_dependency(queue, in_event, func_event);

        queue.wait_and_throw();
        device_free(queue, A_dev);
        device_free(queue, scratchpad_dev);
    }

    return result;
}

InputTestController<decltype(::accuracy<void>)> accuracy_controller{ accuracy_input };
InputTestController<decltype(::usm_dependency<void>)> dependency_controller{ dependency_input };

} /* anonymous namespace */

#include "lapack_gtest_suite.hpp"
INSTANTIATE_GTEST_SUITE_ACCURACY(PotrfBatchStride);
INSTANTIATE_GTEST_SUITE_DEPENDENCY(PotrfBatchStride);
