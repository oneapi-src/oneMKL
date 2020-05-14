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

template <typename fp>
bool test(const device& dev, int N, int incx) {
    // Prepare data.
    vector<fp> x;
    int64_t result = -1, result_ref = -1;
    rand_vector(x, N, incx);

    // Call Reference IAMIN.
    using fp_ref    = typename ref_type_info<fp>::type;
    const int N_ref = N, incx_ref = incx;

    result_ref = ::iamin(&N_ref, (fp_ref*)x.data(), &incx_ref);

    // Call DPC++ IAMIN.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during IAMIN:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(dev, exception_handler);

    buffer<fp, 1> x_buffer = make_buffer(x);
    buffer<int64_t, 1> result_buffer(&result, range<1>(1));

    try {
#ifdef CALL_RT_API
        onemkl::blas::iamin(main_queue, N, x_buffer, incx, result_buffer);
#else
        TEST_RUN_CT(main_queue, onemkl::blas::iamin,
                    (main_queue, N, x_buffer, incx, result_buffer));
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during IAMIN:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good;
    {
        auto result_accessor = result_buffer.template get_access<access::mode::read>();
        good                 = check_equal(result_accessor[0], result_ref, 0, std::cout);
    }

    return good;
}

class IaminTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(IaminTests, RealSinglePrecision) {
    EXPECT_TRUE(test<float>(GetParam(), 1357, 2));
    EXPECT_TRUE(test<float>(GetParam(), 1357, 1));
    EXPECT_TRUE(test<float>(GetParam(), 1357, -3));
}
TEST_P(IaminTests, RealDoublePrecision) {
    EXPECT_TRUE(test<double>(GetParam(), 1357, 2));
    EXPECT_TRUE(test<double>(GetParam(), 1357, 1));
    EXPECT_TRUE(test<double>(GetParam(), 1357, -3));
}
TEST_P(IaminTests, ComplexSinglePrecision) {
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), 1357, 2));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), 1357, 1));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), 1357, -3));
}
TEST_P(IaminTests, ComplexDoublePrecision) {
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), 1357, 2));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), 1357, 1));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), 1357, -3));
}

INSTANTIATE_TEST_SUITE_P(IaminTestSuite, IaminTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
