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
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include <CL/sycl.hpp>
#include "cblas.h"
#include "oneapi/mkl/detail/config.hpp"
#include "oneapi/mkl.hpp"
#include "onemkl_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

using namespace cl::sycl;
using std::vector;

extern std::vector<cl::sycl::device> devices;

namespace {

template <typename fp, typename fp_scalar>
int test(const device &dev, oneapi::mkl::uplo upper_lower, int n, fp_scalar alpha, int incx, int lda) {
    // Prepare data.
    vector<fp> x, A_ref, A;
    rand_vector(x, n, incx);
    rand_matrix(A, oneapi::mkl::transpose::nontrans, n, n, lda);
    A_ref = A;

    // Call Reference HER.
    const int n_ref = n, incx_ref = incx, lda_ref = lda;
    using fp_ref        = typename ref_type_info<fp>::type;
    using fp_scalar_mkl = typename ref_type_info<fp_scalar>::type;

    ::her(convert_to_cblas_uplo(upper_lower), &n_ref, (fp_scalar_mkl *)&alpha, (fp_ref *)x.data(),
          &incx_ref, (fp_ref *)A_ref.data(), &lda_ref);

    // Call DPC++ HER.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during HER:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(dev, exception_handler);

    buffer<fp, 1> x_buffer = make_buffer(x);
    buffer<fp, 1> A_buffer = make_buffer(A);

    try {
#ifdef CALL_RT_API
        oneapi::mkl::blas::her(main_queue, upper_lower, n, alpha, x_buffer, incx, A_buffer, lda);
#else
        TEST_RUN_CT(main_queue, oneapi::mkl::blas::her,
                    (main_queue, upper_lower, n, alpha, x_buffer, incx, A_buffer, lda));
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during HER:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const oneapi::mkl::backend_unsupported_exception &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of HER:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good;
    {
        auto A_accessor = A_buffer.template get_access<access::mode::read>();
        good            = check_equal_matrix(A_accessor, A_ref, n, n, lda, n, std::cout);
    }

    return (int)good;
}

class HerTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(HerTests, ComplexSinglePrecision) {
    float alpha(2.0);
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, float>(GetParam(), oneapi::mkl::uplo::lower, 30, alpha, 2, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, float>(GetParam(), oneapi::mkl::uplo::upper, 30, alpha, 2, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, float>(GetParam(), oneapi::mkl::uplo::lower, 30, alpha, -2, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, float>(GetParam(), oneapi::mkl::uplo::upper, 30, alpha, -2, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, float>(GetParam(), oneapi::mkl::uplo::lower, 30, alpha, 1, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, float>(GetParam(), oneapi::mkl::uplo::upper, 30, alpha, 1, 42)));
}
TEST_P(HerTests, ComplexDoublePrecision) {
    double alpha(2.0);
    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, double>(GetParam(), oneapi::mkl::uplo::lower, 30, alpha, 2, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, double>(GetParam(), oneapi::mkl::uplo::upper, 30, alpha, 2, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, double>(GetParam(), oneapi::mkl::uplo::lower, 30, alpha, -2, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, double>(GetParam(), oneapi::mkl::uplo::upper, 30, alpha, -2, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, double>(GetParam(), oneapi::mkl::uplo::lower, 30, alpha, 1, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, double>(GetParam(), oneapi::mkl::uplo::upper, 30, alpha, 1, 42)));
}

INSTANTIATE_TEST_SUITE_P(HerTestSuite, HerTests, ::testing::ValuesIn(devices), ::DeviceNamePrint());

} // anonymous namespace
