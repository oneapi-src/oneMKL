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
0 29 25 25 30 27182
0 29 25 20 30 27182
1 25 29 25 30 27182
1 25 29 20 30 27182
)";

template <typename mem_T>
bool accuracy(const sycl::device &dev, oneapi::mkl::generate vect, int64_t m, int64_t n, int64_t k, int64_t lda, uint64_t seed) {
    using fp = typename mem_T_info<mem_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    int64_t m_A = m;
    int64_t n_A = n;

    if ( vect == oneapi::mkl::generate::Q )
        n_A = k;
    else /* vect == oneapi::mkl::generate::P */
        m_A = k;

    int64_t min_mn_A = std::min<int64_t>(m_A, n_A);

    std::vector<fp> A(lda*n);
    std::vector<fp_real> d(min_mn_A);
    std::vector<fp_real> e(min_mn_A-1);
    std::vector<fp> tauq(min_mn_A);
    std::vector<fp> taup(min_mn_A);

    rand_matrix(seed, oneapi::mkl::transpose::nontrans, m_A, n_A, A, lda);
    reference::gebrd(m_A, n_A, A.data(), lda, d.data(), e.data(), tauq.data(), taup.data());

    auto& tau = (vect == oneapi::mkl::generate::Q)? tauq: taup;

    /* Compute on device */
    {
        sycl::queue queue{dev};

        auto A_dev = device_alloc<mem_T>(queue, A.size());
        auto tau_dev = device_alloc<mem_T>(queue, tau.size());

#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::ungbr_scratchpad_size<fp>(queue, vect, m, n, k, lda);
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue, scratchpad_size = oneapi::mkl::lapack::ungbr_scratchpad_size<fp>, vect, m, n, k, lda);
#endif
        auto scratchpad_dev = device_alloc<mem_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, tau.data(), tau_dev, tau.size());
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::mkl::lapack::ungbr(queue, vect, m, n, k, A_dev, lda, tau_dev, scratchpad_dev, scratchpad_size);
#else
        TEST_RUN_CT_SELECT(queue, oneapi::mkl::lapack::ungbr, vect, m, n, k, A_dev, lda, tau_dev, scratchpad_dev, scratchpad_size);
#endif
        queue.wait_and_throw();

        device_to_host_copy(queue, A_dev, A.data(), A.size());
        queue.wait_and_throw();

        device_free(queue, A_dev);
        device_free(queue, tau_dev);
        device_free(queue, scratchpad_dev);
    }

    return check_or_un_gbr_accuracy(vect, m, n, k, A.data(), lda);
}

static const char* dependency_input = R"(
1 1 1 1 1 1
)";

template <typename mem_T>
bool usm_dependency(const sycl::device &dev, oneapi::mkl::generate vect, int64_t m, int64_t n, int64_t k, int64_t lda, uint64_t seed) {
    using fp = typename mem_T_info<mem_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    int64_t m_A = m;
    int64_t n_A = n;

    if ( vect == oneapi::mkl::generate::Q )
        n_A = k;
    else /* vect == oneapi::mkl::generate::P */
        m_A = k;

    int64_t min_mn_A = std::min<int64_t>(m_A, n_A);

    std::vector<fp> A(lda*n);
    std::vector<fp_real> d(min_mn_A);
    std::vector<fp_real> e(min_mn_A-1);
    std::vector<fp> tauq(min_mn_A);
    std::vector<fp> taup(min_mn_A);

    rand_matrix(seed, oneapi::mkl::transpose::nontrans, m_A, n_A, A, lda);
    reference::gebrd(m_A, n_A, A.data(), lda, d.data(), e.data(), tauq.data(), taup.data());

    auto& tau = (vect == oneapi::mkl::generate::Q)? tauq: taup;

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{dev};

        auto A_dev = device_alloc<mem_T>(queue, A.size());
        auto tau_dev = device_alloc<mem_T>(queue, tau.size());

#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::ungbr_scratchpad_size<fp>(queue, vect, m, n, k, lda);
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue, scratchpad_size = oneapi::mkl::lapack::ungbr_scratchpad_size<fp>, vect, m, n, k, lda);
#endif
        auto scratchpad_dev = device_alloc<mem_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, tau.data(), tau_dev, tau.size());
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependent_event(queue);
#ifdef CALL_RT_API
        sycl::event func_event = oneapi::mkl::lapack::ungbr(queue, vect, m, n, k, A_dev, lda, tau_dev, scratchpad_dev, scratchpad_size, sycl::vector_class<sycl::event>{in_event});
#else
        sycl::event func_event;
        TEST_RUN_CT_SELECT(queue, sycl::event func_event = oneapi::mkl::lapack::ungbr, vect, m, n, k, A_dev, lda, tau_dev, scratchpad_dev, scratchpad_size, sycl::vector_class<sycl::event>{in_event});
#endif
        result = check_dependency(in_event, func_event);

        queue.wait_and_throw();
        device_free(queue, A_dev);
        device_free(queue, tau_dev);
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
class UngbrTests : public ::testing::TestWithParam<sycl::device*> {};
INSTANTIATE_TEST_SUITE_P(UngbrTestSuite, UngbrTests, ::testing::ValuesIn(devices), DeviceNamePrint());
RUN_SUITE_COMPLEX(Ungbr)
#endif
