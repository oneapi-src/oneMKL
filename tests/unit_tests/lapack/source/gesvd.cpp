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
1 1 8 8 10 10 10 27182
1 1 30 24 42 33 33 27182
)";

template <typename mem_T>
bool accuracy(const sycl::device &dev, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, int64_t m, int64_t n, int64_t lda, int64_t ldu, int64_t ldvt, uint64_t seed) {
    using fp = typename mem_T_info<mem_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    int64_t min_mn = std::min(m, n);
    int64_t ucols = min_mn;
    if (jobu == oneapi::mkl::jobsvd::vectors)
        ucols = m;
    int64_t vtrows = min_mn;
    if (jobvt == oneapi::mkl::jobsvd::vectors)
        vtrows = n;

    std::vector<fp> A(lda*n);
    std::vector<fp> U(ldu*ucols);
    std::vector<fp> Vt(ldvt*n);
    std::vector<fp_real> s(min_mn);

    rand_matrix(seed, oneapi::mkl::transpose::nontrans, m, n, A, lda);
    std::vector<fp> A_initial = A;

    /* Compute on device */
    {
        sycl::queue queue{dev};
        auto A_dev = device_alloc<mem_T>(queue, A.size());
        auto U_dev = device_alloc<mem_T>(queue, U.size());
        auto Vt_dev = device_alloc<mem_T>(queue, Vt.size());
        auto s_dev = device_alloc<mem_T, fp_real>(queue, s.size());
#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::gesvd_scratchpad_size<fp>(queue, jobu, jobvt, m, n, lda, ldu, ldvt);
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue, scratchpad_size = oneapi::mkl::lapack::gesvd_scratchpad_size<fp>, jobu, jobvt, m, n, lda, ldu, ldvt);
#endif
        auto scratchpad_dev = device_alloc<mem_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::mkl::lapack::gesvd(queue, jobu, jobvt, m, n, A_dev, lda, s_dev, U_dev, ldu, Vt_dev, ldvt, scratchpad_dev, scratchpad_size);
#else
        TEST_RUN_CT_SELECT(queue, oneapi::mkl::lapack::gesvd, jobu, jobvt, m, n, A_dev, lda, s_dev, U_dev, ldu, Vt_dev, ldvt, scratchpad_dev, scratchpad_size);
#endif
        queue.wait_and_throw();

        device_to_host_copy(queue, A_dev, A.data(), A.size());
        device_to_host_copy(queue, U_dev, U.data(), U.size());
        device_to_host_copy(queue, Vt_dev, Vt.data(), Vt.size());
        device_to_host_copy(queue, s_dev, s.data(), s.size());
        queue.wait_and_throw();

        device_free(queue, A_dev);
        device_free(queue, U_dev);
        device_free(queue, Vt_dev);
        device_free(queue, s_dev);
        device_free(queue, scratchpad_dev);
    }
    bool result = true;

    if  (jobu == oneapi::mkl::jobsvd::vectors && jobvt == oneapi::mkl::jobsvd::vectors) {
        /* |A - U S V'| < |A| O(eps) */
        std::vector<fp> US(m*n);
        int64_t ldus = m;
        for (int64_t col = 0; col < min_mn; col++)
            for (int64_t row = 0; row < m; row++)
                US[row + col*ldus] = U[row + col*ldu]*s[col];
        std::vector<fp> USV(m*n);
        int64_t ldusv = m;
        reference::gemm(oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, m, n, n, 1.0, US.data(), ldus, Vt.data(), ldvt, 0.0, USV.data(), ldusv);
        if (!rel_mat_err_check(m, n, A_initial.data(), lda, USV.data(), ldusv)) {
            global::log << "\tFactorization check failed" << std::endl;
            result = false;
        }
    }

    if  (jobu == oneapi::mkl::jobsvd::vectorsina)
        reference::lacpy('A', m, ucols, A.data(), lda, U.data(), ldu);
    if  (jobvt == oneapi::mkl::jobsvd::vectorsina)
        reference::lacpy('A', vtrows, n, A.data(), lda, Vt.data(), ldvt);

    if  (jobu == oneapi::mkl::jobsvd::vectors || jobu == oneapi::mkl::jobsvd::somevec || jobu == oneapi::mkl::jobsvd::vectorsina) {
        /* |I - U' U| < n O(eps) */
        std::vector<fp> UU(ucols*ucols);
        int64_t lduu = ucols;
        reference::gemm(oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::nontrans, ucols, ucols, m, 1.0, U.data(), ldu, U.data(), ldu, 0.0, UU.data(), lduu);
        if (!rel_id_err_check(ucols, UU.data(), lduu)) {
            global::log << "\tU Orthogonality check failed" << std::endl;
            result = false;
        }
    }

    if  (jobvt == oneapi::mkl::jobsvd::vectors || jobvt == oneapi::mkl::jobsvd::somevec || jobvt == oneapi::mkl::jobsvd::vectorsina) {
        /* |I - V' V| < n O(eps) */
        std::vector<fp> VV(vtrows*vtrows);
        int64_t ldvv = vtrows;
        reference::gemm(oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::conjtrans, vtrows, vtrows, n, 1.0, Vt.data(), ldvt, Vt.data(), ldvt, 0.0, VV.data(), ldvv);
        if (!rel_id_err_check(vtrows, VV.data(), ldvv)) {
            global::log << "\tV Orthogonality check failed" << std::endl;
            result = false;
        }
    }
    return result;
}

const char* dependency_input = R"(
1 1 1 1 1 1 1 1
)";

template <typename mem_T>
bool usm_dependency(const sycl::device &dev, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, int64_t m, int64_t n, int64_t lda, int64_t ldu, int64_t ldvt, uint64_t seed) {
    using fp = typename mem_T_info<mem_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    int64_t min_mn = std::min(m, n);
    int64_t ucols = min_mn;
    if  (jobu == oneapi::mkl::jobsvd::vectors)
        ucols = m;
    int64_t vtrows = min_mn;
    if(jobvt == oneapi::mkl::jobsvd::vectors)
        vtrows = n;

    std::vector<fp> A(lda*n);
    std::vector<fp> U(ldu*ucols);
    std::vector<fp> Vt(ldvt*n);
    std::vector<fp_real> s(min_mn);

    rand_matrix(seed, oneapi::mkl::transpose::nontrans, m, n, A, lda);
    std::vector<fp> A_initial = A;

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{dev};
        auto A_dev = device_alloc<mem_T>(queue, A.size());
        auto U_dev = device_alloc<mem_T>(queue, U.size());
        auto Vt_dev = device_alloc<mem_T>(queue, Vt.size());
        auto s_dev = device_alloc<mem_T, fp_real>(queue, s.size());
#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::gesvd_scratchpad_size<fp>(queue, jobu, jobvt, m, n, lda, ldu, ldvt);
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue, scratchpad_size = oneapi::mkl::lapack::gesvd_scratchpad_size<fp>, jobu, jobvt, m, n, lda, ldu, ldvt);
#endif
        auto scratchpad_dev = device_alloc<mem_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependent_event(queue);
#ifdef CALL_RT_API
        sycl::event func_event = oneapi::mkl::lapack::gesvd(queue, jobu, jobvt, m, n, A_dev, lda, s_dev, U_dev, ldu, Vt_dev, ldvt, scratchpad_dev, scratchpad_size, sycl::vector_class<sycl::event>{in_event});
#else
        sycl::event func_event;
        TEST_RUN_CT_SELECT(queue, sycl::event func_event = oneapi::mkl::lapack::gesvd, jobu, jobvt, m, n, A_dev, lda, s_dev, U_dev, ldu, Vt_dev, ldvt, scratchpad_dev, scratchpad_size, sycl::vector_class<sycl::event>{in_event});
#endif
        result = check_dependency(in_event, func_event);

        queue.wait_and_throw();
        device_free(queue, A_dev);
        device_free(queue, U_dev);
        device_free(queue, Vt_dev);
        device_free(queue, s_dev);
        device_free(queue, scratchpad_dev);
    }

    return result;
}

InputTestController<decltype(::accuracy<void>)> accuracy_controller{accuracy_input};
InputTestController<decltype(::usm_dependency<void>)> dependency_controller{dependency_input};

} /* unnamed namespace */

#ifdef STANDALONE
int main() {
    sycl::device dev = sycl::device { sycl::host_selector{} };
    int64_t res = 0;
    res += !accuracy_controller.run(::accuracy<RealSinglePrecisionUsm>, dev);
    res += !accuracy_controller.run(::accuracy<RealDoublePrecisionUsm>, dev);
    res += !accuracy_controller.run(::accuracy<ComplexSinglePrecisionUsm>, dev);
    res += !accuracy_controller.run(::accuracy<ComplexDoublePrecisionUsm>, dev);
    res += !accuracy_controller.run(::accuracy<RealSinglePrecisionBuffer>, dev);
    res += !accuracy_controller.run(::accuracy<RealDoublePrecisionBuffer>, dev);
    res += !accuracy_controller.run(::accuracy<ComplexSinglePrecisionBuffer>, dev);
    res += !accuracy_controller.run(::accuracy<ComplexDoublePrecisionBuffer>, dev);
    res += !dependency_controller.run(::usm_dependency<RealSinglePrecisionUsm>, dev);
    res += !dependency_controller.run(::usm_dependency<RealDoublePrecisionUsm>, dev);
    res += !dependency_controller.run(::usm_dependency<ComplexSinglePrecisionUsm>, dev);
    res += !dependency_controller.run(::usm_dependency<ComplexDoublePrecisionUsm>, dev);
    return res;
}
#else
#include <gtest/gtest.h>
extern std::vector<sycl::device*> devices;
class GesvdTests : public ::testing::TestWithParam<sycl::device*> {};
INSTANTIATE_TEST_SUITE_P(GesvdTestSuite, GesvdTests, ::testing::ValuesIn(devices), DeviceNamePrint());
RUN_SUITE(Gesvd)
#endif
