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
1 0 0 25 79 66 38 27182
1 0 1 32 34 92 39 27182
1 3 0 76 61 87 82 27182
1 3 1 89 92 89 99 27182
0 0 0 25 79 66 38 27182
0 0 1 32 34 92 39 27182
0 3 0 76 61 87 82 27182
0 3 1 89 92 89 99 27182
)";

template <typename mem_T>
bool accuracy(const sycl::device &dev, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, oneapi::mkl::diag diag, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb, uint64_t seed ) {
    using fp = typename mem_T_info<mem_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A(lda*n);
    std::vector<fp> B(ldb*nrhs);

    /* Initialize input data */
    rand_matrix(seed, oneapi::mkl::transpose::nontrans, n, n, A, lda);
    rand_matrix(seed, oneapi::mkl::transpose::nontrans, n, nrhs, B, ldb);
    std::vector<fp> B_initial = B;

    /* Compute on device */
    {
        sycl::queue queue{dev};

        auto A_dev = device_alloc<mem_T>(queue, A.size());
        auto B_dev = device_alloc<mem_T>(queue, B.size());

#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::trtrs_scratchpad_size<fp>(queue, uplo, trans, diag, n, nrhs, lda, ldb);
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue, scratchpad_size = oneapi::mkl::lapack::trtrs_scratchpad_size<fp>, uplo, trans, diag, n, nrhs, lda, ldb);
#endif
        auto scratchpad_dev = device_alloc<mem_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, B.data(), B_dev, B.size());
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::mkl::lapack::trtrs(queue, uplo, trans, diag, n, nrhs, A_dev, lda, B_dev, ldb, scratchpad_dev, scratchpad_size);
#else
        TEST_RUN_CT_SELECT(queue, oneapi::mkl::lapack::trtrs, uplo, trans, diag, n, nrhs, A_dev, lda, B_dev, ldb, scratchpad_dev, scratchpad_size);
#endif
        queue.wait_and_throw();

        device_to_host_copy(queue, B_dev, B.data(), B.size());
        queue.wait_and_throw();

        device_free(queue, A_dev);
        device_free(queue, B_dev);
        device_free(queue, scratchpad_dev);
    }

    return check_trtrs_accuracy(uplo, trans, diag, n, nrhs, A.data(), lda, B.data(), ldb, B_initial.data());
}

static const char* dependency_input = R"(
1 1 1 1 1 1 1 1
)";

template <typename mem_T>
bool usm_dependency(const sycl::device &dev, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, oneapi::mkl::diag diag, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb, uint64_t seed) {
    using fp = typename mem_T_info<mem_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A(lda*n);
    std::vector<fp> B(ldb*nrhs);

    /* Initialize input data */
    rand_matrix(seed, oneapi::mkl::transpose::nontrans, n, n, A, lda);
    rand_matrix(seed, oneapi::mkl::transpose::nontrans, n, nrhs, B, ldb);
    std::vector<fp> B_initial = B;

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{dev};

        auto A_dev = device_alloc<mem_T>(queue, A.size());
        auto B_dev = device_alloc<mem_T>(queue, B.size());

#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::trtrs_scratchpad_size<fp>(queue, uplo, trans, diag, n, nrhs, lda, ldb);
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue, scratchpad_size = oneapi::mkl::lapack::trtrs_scratchpad_size<fp>, uplo, trans, diag, n, nrhs, lda, ldb);
#endif
        auto scratchpad_dev = device_alloc<mem_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, B.data(), B_dev, B.size());
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependent_event(queue);
#ifdef CALL_RT_API
        sycl::event func_event = oneapi::mkl::lapack::trtrs(queue, uplo, trans, diag, n, nrhs, A_dev, lda, B_dev, ldb, scratchpad_dev, scratchpad_size, sycl::vector_class<sycl::event>{in_event});
#else
        sycl::event func_event;
        TEST_RUN_CT_SELECT(queue, sycl::event func_event = oneapi::mkl::lapack::trtrs, uplo, trans, diag, n, nrhs, A_dev, lda, B_dev, ldb, scratchpad_dev, scratchpad_size, sycl::vector_class<sycl::event>{in_event});
#endif
        result = check_dependency(in_event, func_event);

        queue.wait_and_throw();
        device_free(queue, A_dev);
        device_free(queue, B_dev);
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
class TrtrsTests : public ::testing::TestWithParam<sycl::device*> {};
INSTANTIATE_TEST_SUITE_P(TrtrsTestSuite, TrtrsTests, ::testing::ValuesIn(devices), DeviceNamePrint());
RUN_SUITE(Trtrs)
#endif
