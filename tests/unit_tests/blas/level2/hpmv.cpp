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
#include "config.hpp"
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
bool test(const device &dev, onemkl::uplo upper_lower, int n, fp alpha, fp beta, int incx,
          int incy) {
    // Prepare data.
    vector<fp> x, y, y_ref, A;
    rand_vector(x, n, incx);
    rand_vector(y, n, incy);
    y_ref = y;
    rand_matrix(A, onemkl::transpose::nontrans, n, n, n);

    // Call Reference HPMV.
    const int n_ref = n, incx_ref = incx, incy_ref = incy;
    using fp_ref = typename ref_type_info<fp>::type;

    ::hpmv(convert_to_cblas_uplo(upper_lower), &n_ref, (fp_ref *)&alpha, (fp_ref *)A.data(),
           (fp_ref *)x.data(), &incx_ref, (fp_ref *)&beta, (fp_ref *)y_ref.data(), &incy_ref);

    // Call DPC++ HPMV.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during HPMV:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(dev, exception_handler);

    buffer<fp, 1> x_buffer = make_buffer(x);
    buffer<fp, 1> y_buffer = make_buffer(y);
    buffer<fp, 1> A_buffer = make_buffer(A);

    try {
#ifdef CALL_RT_API
        onemkl::blas::hpmv(main_queue, upper_lower, n, alpha, A_buffer, x_buffer, incx, beta,
                           y_buffer, incy);
#else
        TEST_RUN_CT(
            main_queue, onemkl::blas::hpmv,
            (main_queue, upper_lower, n, alpha, A_buffer, x_buffer, incx, beta, y_buffer, incy));
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during HPMV:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good;
    {
        auto y_accessor = y_buffer.template get_access<access::mode::read>();
        good            = check_equal_vector(y_accessor, y_ref, n, incy, n, std::cout);
    }

    return good;
}

class HpmvTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(HpmvTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0, -0.5);
    std::complex<float> beta(3.0, -1.5);
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::uplo::lower, 30, alpha, beta, 2, 3));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::uplo::upper, 30, alpha, beta, 2, 3));
    EXPECT_TRUE(
        test<std::complex<float>>(GetParam(), onemkl::uplo::lower, 30, alpha, beta, -2, -3));
    EXPECT_TRUE(
        test<std::complex<float>>(GetParam(), onemkl::uplo::upper, 30, alpha, beta, -2, -3));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::uplo::lower, 30, alpha, beta, 1, 1));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::uplo::upper, 30, alpha, beta, 1, 1));
}
TEST_P(HpmvTests, ComplexDoublePrecision) {
    std::complex<double> alpha(2.0, -0.5);
    std::complex<double> beta(3.0, -1.5);
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::uplo::lower, 30, alpha, beta, 2, 3));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::uplo::upper, 30, alpha, beta, 2, 3));
    EXPECT_TRUE(
        test<std::complex<double>>(GetParam(), onemkl::uplo::lower, 30, alpha, beta, -2, -3));
    EXPECT_TRUE(
        test<std::complex<double>>(GetParam(), onemkl::uplo::upper, 30, alpha, beta, -2, -3));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::uplo::lower, 30, alpha, beta, 1, 1));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::uplo::upper, 30, alpha, beta, 1, 1));
}

INSTANTIATE_TEST_SUITE_P(HpmvTestSuite, HpmvTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
