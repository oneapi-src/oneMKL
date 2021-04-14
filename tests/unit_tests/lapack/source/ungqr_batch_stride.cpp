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
29 23 18 37 1024 40 3 27182
)";

template <typename mem_T>
bool accuracy(const sycl::device& dev, int64_t m, int64_t n, int64_t k, int64_t lda,
              int64_t stride_a, int64_t stride_tau, int64_t batch_size, uint64_t seed) {
    using fp = typename mem_T_info<mem_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A(stride_a * batch_size);
    std::vector<fp> tau(stride_tau * batch_size);

    for (int64_t i = 0; i < batch_size; i++) {
        rand_matrix(seed, oneapi::mkl::transpose::nontrans, m, n, A, lda, i * stride_a);
        auto info =
            reference::geqrf(m, k, A.data() + i * stride_a, lda, tau.data() + i * stride_tau);
        if (0 != info) {
            global::log << "\tbatch routine index " << i
                        << ": reference geqrf failed with info: " << info << std::endl;
            return false;
        }
    }

    /* Compute on device */
    {
        sycl::queue queue{ dev };

        auto A_dev = device_alloc<mem_T>(queue, A.size());
        auto tau_dev = device_alloc<mem_T>(queue, tau.size());
#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::ungqr_batch_scratchpad_size<fp>(
            queue, m, n, k, lda, stride_a, stride_tau, batch_size);
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue,
                           scratchpad_size = oneapi::mkl::lapack::ungqr_batch_scratchpad_size<fp>,
                           m, n, k, lda, stride_a, stride_tau, batch_size);
#endif
        auto scratchpad_dev = device_alloc<mem_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, tau.data(), tau_dev, tau.size());
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::mkl::lapack::ungqr_batch(queue, m, n, k, A_dev, lda, stride_a, tau_dev, stride_tau,
                                         batch_size, scratchpad_dev, scratchpad_size);
#else
        TEST_RUN_CT_SELECT(queue, oneapi::mkl::lapack::ungqr_batch, m, n, k, A_dev, lda, stride_a,
                           tau_dev, stride_tau, batch_size, scratchpad_dev, scratchpad_size);
#endif
        queue.wait_and_throw();

        device_to_host_copy(queue, A_dev, A.data(), A.size());
        device_to_host_copy(queue, tau_dev, tau.data(), tau.size());
        queue.wait_and_throw();

        device_free(queue, A_dev);
        device_free(queue, tau_dev);
        device_free(queue, scratchpad_dev);
    }

    bool result = true;
    for (int64_t i = 0; i < batch_size; i++)
        if (!check_or_un_gqr_accuracy(m, n, A.data() + i * stride_a, lda)) {
            global::log << "\tbatch routine index " << i << " failed" << std::endl;
            result = false;
        }

    return result;
}

const char* dependency_input = R"(
1 1 1 1 1 1 1 1
)";

template <typename mem_T>
bool usm_dependency(const sycl::device& dev, int64_t m, int64_t n, int64_t k, int64_t lda,
                    int64_t stride_a, int64_t stride_tau, int64_t batch_size, uint64_t seed) {
    using fp = typename mem_T_info<mem_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A(stride_a * batch_size);
    std::vector<fp> tau(stride_tau * batch_size);

    for (int64_t i = 0; i < batch_size; i++) {
        rand_matrix(seed, oneapi::mkl::transpose::nontrans, m, n, A, lda, i * stride_a);
        auto info =
            reference::geqrf(m, k, A.data() + i * stride_a, lda, tau.data() + i * stride_tau);
        if (0 != info) {
            global::log << "\tbatch routine index " << i
                        << ": reference geqrf failed with info: " << info << std::endl;
            return false;
        }
    }

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{ dev };

        auto A_dev = device_alloc<mem_T>(queue, A.size());
        auto tau_dev = device_alloc<mem_T>(queue, tau.size());
#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::ungqr_batch_scratchpad_size<fp>(
            queue, m, n, k, lda, stride_a, stride_tau, batch_size);
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue,
                           scratchpad_size = oneapi::mkl::lapack::ungqr_batch_scratchpad_size<fp>,
                           m, n, k, lda, stride_a, stride_tau, batch_size);
#endif
        auto scratchpad_dev = device_alloc<mem_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, tau.data(), tau_dev, tau.size());
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependent_event(queue);
#ifdef CALL_RT_API
        sycl::event func_event = oneapi::mkl::lapack::ungqr_batch(
            queue, m, n, k, A_dev, lda, stride_a, tau_dev, stride_tau, batch_size, scratchpad_dev,
            scratchpad_size, sycl::vector_class<sycl::event>{ in_event });
#else
        sycl::event func_event;
        TEST_RUN_CT_SELECT(queue, sycl::event func_event = oneapi::mkl::lapack::ungqr_batch, m, n,
                           k, A_dev, lda, stride_a, tau_dev, stride_tau, batch_size, scratchpad_dev,
                           scratchpad_size, sycl::vector_class<sycl::event>{ in_event });
#endif
        result = check_dependency(queue, in_event, func_event);

        queue.wait_and_throw();
        device_free(queue, A_dev);
        device_free(queue, tau_dev);
        device_free(queue, scratchpad_dev);
    }

    return result;
}

InputTestController<decltype(::accuracy<void>)> accuracy_controller{ accuracy_input };
InputTestController<decltype(::usm_dependency<void>)> dependency_controller{ dependency_input };

} /* unnamed namespace */

#ifdef STANDALONE
int main() {
    sycl::device dev = sycl::device{ sycl::host_selector{} };
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
class UngqrBatchStrideTests : public ::testing::TestWithParam<sycl::device*> {};
INSTANTIATE_TEST_SUITE_P(UngqrBatchStrideTestSuite, UngqrBatchStrideTests,
                         ::testing::ValuesIn(devices), DeviceNamePrint());
RUN_SUITE_COMPLEX(UngqrBatchStride)
#endif
