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

template <typename fp>
int test(const device &dev, int N, int incx, int incy, fp flag) {
    // Prepare data.
    vector<fp> x, x_ref, y, y_ref;
    vector<fp> param;
    rand_vector(x, N, incx);
    rand_vector(y, N, incy);
    rand_vector(param, 5, 1);
    param[0] = flag;
    y_ref = y;
    x_ref = x;

    // Call Reference ROTM.
    using fp_ref = typename ref_type_info<fp>::type;
    const int N_ref = N, incx_ref = incx, incy_ref = incy;

    ::rotm(&N_ref, (fp_ref *)x_ref.data(), &incx_ref, (fp_ref *)y_ref.data(), &incy_ref,
           (fp_ref *)param.data());

    // Call DPC++ ROTM.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during ROTM:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(dev, exception_handler);

    buffer<fp, 1> x_buffer = make_buffer(x);
    buffer<fp, 1> y_buffer = make_buffer(y);
    buffer<fp, 1> param_buffer = make_buffer(param);

    try {
#ifdef CALL_RT_API
        oneapi::mkl::blas::rotm(main_queue, N, x_buffer, incx, y_buffer, incy, param_buffer);
#else
        TEST_RUN_CT(main_queue, oneapi::mkl::blas::rotm,
                    (main_queue, N, x_buffer, incx, y_buffer, incy, param_buffer));
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during ROTM:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const oneapi::mkl::backend_unsupported_exception &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of ROTM:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good;
    {
        auto x_accessor = x_buffer.template get_access<access::mode::read>();
        bool good_x = check_equal_vector(x_accessor, x_ref, N, incx, N, std::cout);
        auto y_accessor = y_buffer.template get_access<access::mode::read>();
        bool good_y = check_equal_vector(y_accessor, y_ref, N, incy, N, std::cout);
        good = good_x && good_y;
    }

    return (int)good;
}

class RotmTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(RotmTests, RealSinglePrecision) {
    float flag(-1.0);
    EXPECT_TRUEORSKIP(test<float>(GetParam(), 1357, 2, 3, flag));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), 1357, -2, -3, flag));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), 1357, 1, 1, flag));
    flag = 0.0;
    EXPECT_TRUEORSKIP(test<float>(GetParam(), 1357, 2, 3, flag));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), 1357, -2, -3, flag));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), 1357, 1, 1, flag));
    flag = 1.0;
    EXPECT_TRUEORSKIP(test<float>(GetParam(), 1357, 2, 3, flag));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), 1357, -2, -3, flag));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), 1357, 1, 1, flag));
    flag = -2.0;
    EXPECT_TRUEORSKIP(test<float>(GetParam(), 1357, 2, 3, flag));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), 1357, -2, -3, flag));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), 1357, 1, 1, flag));
}
TEST_P(RotmTests, RealDoublePrecision) {
    double flag(-1.0);
    EXPECT_TRUEORSKIP(test<double>(GetParam(), 1357, 2, 3, flag));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), 1357, -2, -3, flag));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), 1357, 1, 1, flag));
    flag = 0.0;
    EXPECT_TRUEORSKIP(test<double>(GetParam(), 1357, 2, 3, flag));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), 1357, -2, -3, flag));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), 1357, 1, 1, flag));
    flag = 1.0;
    EXPECT_TRUEORSKIP(test<double>(GetParam(), 1357, 2, 3, flag));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), 1357, -2, -3, flag));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), 1357, 1, 1, flag));
    flag = -2.0;
    EXPECT_TRUEORSKIP(test<double>(GetParam(), 1357, 2, 3, flag));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), 1357, -2, -3, flag));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), 1357, 1, 1, flag));
}

INSTANTIATE_TEST_SUITE_P(RotmTestSuite, RotmTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
