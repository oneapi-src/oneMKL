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
1 0 30 30 30 33 31 27182
)";

template <typename mem_T>
bool accuracy(const sycl::device& dev, oneapi::mkl::side left_right, oneapi::mkl::transpose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc, uint64_t seed) {
    using fp = typename mem_T_info<mem_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> C_initial(ldc*n);
    rand_matrix(seed, oneapi::mkl::transpose::nontrans, m, n, C_initial, ldc);
    std::vector<fp> C = C_initial;

    int64_t nq;
    if (left_right == oneapi::mkl::side::left) {
        if ( k > m ) {
            global::log << "\tBad test input, side == left and k > m (" << k << " > " << m << ")" << std::endl;
            return false;
        }
        nq = m;
    } else {
        if ( k > n ) {
            global::log << "\tBad test input, side == right and k > n (" << k << " > " << n << ")" << std::endl;
            return false;
        }
        nq = n;
    }

    std::vector<fp> A(lda*k);
    rand_matrix(seed, oneapi::mkl::transpose::nontrans, nq, k, A, lda);
    std::vector<fp> tau(k);

    auto info = reference::gerqf(nq, k, A.data(), lda, tau.data());
    if( 0 != info) {
        global::log << "\treference gerqf failed with info = " << info << std::endl;
        return false;
    }

    /* Compute on device */
    {
        sycl::queue queue{dev};
        auto A_dev   = device_alloc<mem_T>(queue, A.size());
        auto tau_dev = device_alloc<mem_T>(queue, tau.size());
        auto C_dev   = device_alloc<mem_T>(queue, C.size());
#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::unmrq_scratchpad_size<fp>(queue, left_right, trans, m, n, k, lda, ldc);
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue, scratchpad_size = oneapi::mkl::lapack::unmrq_scratchpad_size<fp>, left_right, trans, m, n, k, lda, ldc);
#endif
        auto scratchpad_dev = device_alloc<mem_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, tau.data(), tau_dev, tau.size());
        host_to_device_copy(queue, C.data(), C_dev, C.size());
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::mkl::lapack::unmrq(queue, left_right, trans, m, n, k, A_dev, lda, tau_dev, C_dev, ldc, scratchpad_dev, scratchpad_size);
#else
        TEST_RUN_CT_SELECT(queue, oneapi::mkl::lapack::unmrq, left_right, trans, m, n, k, A_dev, lda, tau_dev, C_dev, ldc, scratchpad_dev, scratchpad_size);
#endif
        queue.wait_and_throw();

        device_to_host_copy(queue, C_dev, C.data(), C.size());
        queue.wait_and_throw();

        device_free(queue, A_dev);
        device_free(queue, tau_dev);
        device_free(queue, C_dev);
        device_free(queue, scratchpad_dev);
    }
    bool result = true;

    /* |Q C - QC| < |QC| O(eps) */
    const auto& QC = C;
    auto& QC_ref = C_initial;
    auto ldqc = ldc;
    info = reference::or_un_mrq(left_right, trans, m, n, k, A.data(), lda, tau.data(), QC_ref.data(), ldqc);
    if ( 0 != info) {
        global::log << "\treference unmrq failed with info = " << info << std::endl;
        return false;
    }
    if(!rel_mat_err_check(m, n, QC.data(), ldqc, QC_ref.data(), ldqc, 1.0)) {
        global::log << "\tMultiplication check failed" << std::endl;
        result = false;
    }
    return result;
}

static const char* dependency_input = R"(
1 1 1 1 1 1 1 1
)";

template <typename mem_T>
bool usm_dependency(const sycl::device& dev, oneapi::mkl::side left_right, oneapi::mkl::transpose trans, int64_t m, int64_t n, int64_t k , int64_t lda, int64_t ldc, uint64_t seed) {
    using fp = typename mem_T_info<mem_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> C_initial(ldc*n);
    rand_matrix(seed, oneapi::mkl::transpose::nontrans, m, n, C_initial, ldc);
    std::vector<fp> C = C_initial;

    int64_t nq = (left_right==oneapi::mkl::side::left) ? m : n;
    std::vector<fp> A(lda*k);
    rand_matrix(seed, oneapi::mkl::transpose::nontrans, nq, k, A, lda);
    std::vector<fp> tau(k);

    auto info = reference::gerqf(nq, k, A.data(), lda, tau.data());
    if( 0 != info) {
        global::log << "\treference gerqf failed with info = " << info << std::endl;
        return false;
    }

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{dev};
        auto A_dev   = device_alloc<mem_T>(queue, A.size());
        auto tau_dev = device_alloc<mem_T>(queue, tau.size());
        auto C_dev   = device_alloc<mem_T>(queue, C.size());
#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::unmrq_scratchpad_size<fp>(queue, left_right, trans, m, n, k, lda, ldc);
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue, scratchpad_size = oneapi::mkl::lapack::unmrq_scratchpad_size<fp>, left_right, trans, m, n, k, lda, ldc);
#endif
        auto scratchpad_dev = device_alloc<mem_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, tau.data(), tau_dev, tau.size());
        host_to_device_copy(queue, C.data(), C_dev, C.size());
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependent_event(queue);
#ifdef CALL_RT_API
        sycl::event func_event = oneapi::mkl::lapack::unmrq(queue, left_right, trans, m, n, k, A_dev, lda, tau_dev, C_dev, ldc, scratchpad_dev, scratchpad_size, sycl::vector_class<sycl::event>{in_event});
#else
        sycl::event func_event;
        TEST_RUN_CT_SELECT(queue, sycl::event func_event = oneapi::mkl::lapack::unmrq, left_right, trans, m, n, k, A_dev, lda, tau_dev, C_dev, ldc, scratchpad_dev, scratchpad_size, sycl::vector_class<sycl::event>{in_event});
#endif
        result = check_dependency(in_event, func_event);

        queue.wait_and_throw();
        device_free(queue, A_dev);
        device_free(queue, tau_dev);
        device_free(queue, C_dev);
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
class UnmrqTests : public ::testing::TestWithParam<sycl::device*> {};
INSTANTIATE_TEST_SUITE_P(UnmrqTestSuite, UnmrqTests, ::testing::ValuesIn(devices), DeviceNamePrint());
RUN_SUITE_COMPLEX(Unmrq)
#endif
