/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include <CL/sycl.hpp>
#include "cblas.h"
#include "onemkl/detail/config.hpp"
#include "onemkl/onemkl.hpp"
#include "onemkl_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

using namespace cl::sycl;
using std::vector;

extern std::vector<cl::sycl::device> devices;

namespace {

template <typename fp>
int test(const device& dev, onemkl::uplo upper_lower, onemkl::transpose trans, int n, int k,
         int lda, int ldc, fp alpha, fp beta) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during SYRK:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(dev, exception_handler);
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.
    auto ua = usm_allocator<fp, usm::alloc::shared, 64>(cxt, dev);
    vector<fp, decltype(ua)> A(ua), C(ua);
    rand_matrix(A, trans, n, k, lda);
    rand_matrix(C, onemkl::transpose::nontrans, n, n, ldc);

    auto C_ref = C;

    // Call Reference SYRK.
    const int n_ref = n, k_ref = k;
    const int lda_ref = lda, ldc_ref = ldc;

    using fp_ref = typename ref_type_info<fp>::type;

    ::syrk(convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans), &n_ref, &k_ref,
           (fp_ref*)&alpha, (fp_ref*)A.data(), &lda_ref, (fp_ref*)&beta, (fp_ref*)C_ref.data(),
           &ldc_ref);

    // Call DPC++ SYRK.

    try {
#ifdef CALL_RT_API
        done = onemkl::blas::syrk(main_queue, upper_lower, trans, n, k, alpha, A.data(), lda, beta,
                                  C.data(), ldc, dependencies);
        done.wait();
#else
        TEST_RUN_CT(main_queue, onemkl::blas::syrk,
                    (main_queue, upper_lower, trans, n, k, alpha, A.data(), lda, beta, C.data(),
                     ldc, dependencies));
        main_queue.wait();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during SYRK:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const onemkl::backend_unsupported_exception& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of SYRK:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good = check_equal_matrix(C, C_ref, n, n, ldc, 10 * std::max(n, k), std::cout);

    return (int)good;
}

class SyrkUsmTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(SyrkUsmTests, RealSinglePrecision) {
    float alpha(3.0);
    float beta(3.0);
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans, 73,
                                  27, 101, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans, 73,
                                  27, 101, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::lower, onemkl::transpose::trans, 73, 27,
                                  101, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::upper, onemkl::transpose::trans, 73, 27,
                                  101, 103, alpha, beta));
}
TEST_P(SyrkUsmTests, RealDoublePrecision) {
    double alpha(3.0);
    double beta(3.0);
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans, 73,
                                   27, 101, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans, 73,
                                   27, 101, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::lower, onemkl::transpose::trans, 73,
                                   27, 101, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::upper, onemkl::transpose::trans, 73,
                                   27, 101, 103, alpha, beta));
}
TEST_P(SyrkUsmTests, ComplexSinglePrecision) {
    std::complex<float> alpha(3.0, -0.5);
    std::complex<float> beta(3.0, -1.5);
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::uplo::lower,
                                                onemkl::transpose::nontrans, 73, 27, 101, 103,
                                                alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::uplo::upper,
                                                onemkl::transpose::nontrans, 73, 27, 101, 103,
                                                alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::trans, 73, 27, 101, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::trans, 73, 27, 101, 103, alpha, beta));
}
TEST_P(SyrkUsmTests, ComplexDoublePrecision) {
    std::complex<double> alpha(3.0, -0.5);
    std::complex<double> beta(3.0, -1.5);
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::uplo::lower,
                                                 onemkl::transpose::nontrans, 73, 27, 101, 103,
                                                 alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::uplo::upper,
                                                 onemkl::transpose::nontrans, 73, 27, 101, 103,
                                                 alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::trans, 73, 27, 101, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::trans, 73, 27, 101, 103, alpha, beta));
}

INSTANTIATE_TEST_SUITE_P(SyrkUsmTestSuite, SyrkUsmTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
