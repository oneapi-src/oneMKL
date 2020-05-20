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

#include <cstdint>
#include <cstdlib>
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

template <typename fp, typename fp_scalar>
bool test(const device& dev, int N, int incx, fp_scalar alpha) {
    // Prepare data.
    vector<fp> x, x_ref;

    rand_vector(x, N, incx);
    x_ref = x;

    // Call Reference SCAL.
    using fp_ref        = typename ref_type_info<fp>::type;
    using fp_scalar_mkl = typename ref_type_info<fp_scalar>::type;

    const int N_ref = N, incx_ref = std::abs(incx);

    ::scal(&N_ref, (fp_scalar_mkl*)&alpha, (fp_ref*)x_ref.data(), &incx_ref);

    // Call DPC++ SCAL.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during SCAL:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(dev, exception_handler);

    buffer<fp, 1> x_buffer = make_buffer(x);

    try {
#ifdef CALL_RT_API
        onemkl::blas::scal(main_queue, N, alpha, x_buffer, incx);
#else
        TEST_RUN_CT(main_queue, onemkl::blas::scal, (main_queue, N, alpha, x_buffer, incx));
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during SCAL:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good;
    {
        auto x_accessor = x_buffer.template get_access<access::mode::read>();
        good            = check_equal_vector(x_accessor, x_ref, N, incx, N, std::cout);
    }

    return good;
}

class ScalTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(ScalTests, RealSinglePrecision) {
    float alpha(2.0);
    EXPECT_TRUE((test<float, float>(GetParam(), 1357, 2, alpha)));
    EXPECT_TRUE((test<float, float>(GetParam(), 1357, -3, alpha)));
}
TEST_P(ScalTests, RealDoublePrecision) {
    double alpha(2.0);
    EXPECT_TRUE((test<double, double>(GetParam(), 1357, 2, alpha)));
    EXPECT_TRUE((test<double, double>(GetParam(), 1357, -3, alpha)));
}
TEST_P(ScalTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0, -0.5);
    EXPECT_TRUE((test<std::complex<float>, std::complex<float>>(GetParam(), 1357, 2, alpha)));
    EXPECT_TRUE((test<std::complex<float>, std::complex<float>>(GetParam(), 1357, -3, alpha)));
}
TEST_P(ScalTests, ComplexDoublePrecision) {
    std::complex<double> alpha(2.0, -0.5);
    EXPECT_TRUE((test<std::complex<double>, std::complex<double>>(GetParam(), 1357, 2, alpha)));
    EXPECT_TRUE((test<std::complex<double>, std::complex<double>>(GetParam(), 1357, -3, alpha)));
}
TEST_P(ScalTests, ComplexRealSinglePrecision) {
    float alpha(2.0);
    EXPECT_TRUE((test<std::complex<float>, float>(GetParam(), 1357, 2, alpha)));
    EXPECT_TRUE((test<std::complex<float>, float>(GetParam(), 1357, -3, alpha)));
}
TEST_P(ScalTests, ComplexRealDoublePrecision) {
    double alpha(2.0);
    EXPECT_TRUE((test<std::complex<double>, double>(GetParam(), 1357, 2, alpha)));
    EXPECT_TRUE((test<std::complex<double>, double>(GetParam(), 1357, -3, alpha)));
}

INSTANTIATE_TEST_SUITE_P(ScalTestSuite, ScalTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
