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
bool test(const device& dev, int N, int incx, int incy) {
    // Prepare data.
    vector<fp> x, x_ref, y, y_ref;
    rand_vector(x, N, incx);
    rand_vector(y, N, incy);
    y_ref = y;
    x_ref = x;

    // Call Reference SWAP.
    using fp_ref    = typename ref_type_info<fp>::type;
    const int N_ref = N, incx_ref = incx, incy_ref = incy;

    ::swap(&N_ref, (fp_ref*)x_ref.data(), &incx_ref, (fp_ref*)y_ref.data(), &incy_ref);

    // Call DPC++ SWAP.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during SWAP:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(dev, exception_handler);

    buffer<fp, 1> x_buffer = make_buffer(x);
    buffer<fp, 1> y_buffer = make_buffer(y);

    try {
#ifdef CALL_RT_API
        onemkl::blas::swap(main_queue, N, x_buffer, incx, y_buffer, incy);
#else
        TEST_RUN_CT(main_queue, onemkl::blas::swap,
                    (main_queue, N, x_buffer, incx, y_buffer, incy));
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during SWAP:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good;
    {
        auto y_accessor = y_buffer.template get_access<access::mode::read>();
        auto x_accessor = x_buffer.template get_access<access::mode::read>();
        bool good_y     = check_equal_vector(y_accessor, y_ref, N, incy, N, std::cout);
        bool good_x     = check_equal_vector(x_accessor, x_ref, N, incx, N, std::cout);
        good            = good_x && good_y;
    }

    return good;
}

class SwapTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(SwapTests, RealSinglePrecision) {
    EXPECT_TRUE(test<float>(GetParam(), 1357, 2, 3));
    EXPECT_TRUE(test<float>(GetParam(), 1357, -2, -3));
    EXPECT_TRUE(test<float>(GetParam(), 1357, 1, 1));
}
TEST_P(SwapTests, RealDoublePrecision) {
    EXPECT_TRUE(test<double>(GetParam(), 1357, 2, 3));
    EXPECT_TRUE(test<double>(GetParam(), 1357, -2, -3));
    EXPECT_TRUE(test<double>(GetParam(), 1357, 1, 1));
}
TEST_P(SwapTests, ComplexSinglePrecision) {
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), 1357, 2, 3));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), 1357, -2, -3));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), 1357, 1, 1));
}
TEST_P(SwapTests, ComplexDoublePrecision) {
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), 1357, 2, 3));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), 1357, -2, -3));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), 1357, 1, 1));
}

INSTANTIATE_TEST_SUITE_P(SwapTestSuite, SwapTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
