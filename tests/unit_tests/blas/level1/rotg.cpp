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
bool test(const device &dev, fp s, fp_scalar c) {
    // Prepare data.
    fp a, b, a_ref, b_ref, s_ref;
    fp_scalar c_ref;

    a     = rand_scalar<fp>();
    b     = rand_scalar<fp>();
    a_ref = a;
    b_ref = b;

    // Call Reference ROTG.
    using fp_ref = typename ref_type_info<fp>::type;

    ::rotg((fp_ref *)&a_ref, (fp_ref *)&b_ref, (fp_scalar *)&c_ref, (fp_ref *)&s_ref);

    // Call DPC++ ROTG.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during ROTG:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(dev, exception_handler);

    buffer<fp, 1> a_buffer(&a, range<1>(1));
    buffer<fp, 1> b_buffer(&b, range<1>(1));
    buffer<fp_scalar, 1> c_buffer(&c, range<1>(1));
    buffer<fp, 1> s_buffer(&s, range<1>(1));

    try {
#ifdef CALL_RT_API
        onemkl::blas::rotg(main_queue, a_buffer, b_buffer, c_buffer, s_buffer);
#else
        TEST_RUN_CT(main_queue, onemkl::blas::rotg,
                    (main_queue, a_buffer, b_buffer, c_buffer, s_buffer));
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during ROTG:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good;
    {
        auto a_accessor = a_buffer.template get_access<access::mode::read>();
        bool good_a     = check_equal(a_accessor[0], a_ref, 4, std::cout);
        auto b_accessor = b_buffer.template get_access<access::mode::read>();
        bool good_b     = check_equal(b_accessor[0], b_ref, 4, std::cout);
        auto s_accessor = s_buffer.template get_access<access::mode::read>();
        bool good_s     = check_equal(s_accessor[0], s_ref, 4, std::cout);
        auto c_accessor = c_buffer.template get_access<access::mode::read>();
        bool good_c     = check_equal(c_accessor[0], c_ref, 4, std::cout);

        good = good_a && good_b && good_c && good_s;
    }

    return good;
}

class RotgTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(RotgTests, RealSinglePrecision) {
    float c(2.0);
    float s(-0.5);
    EXPECT_TRUE((test<float, float>(GetParam(), c, s)));
    EXPECT_TRUE((test<float, float>(GetParam(), c, s)));
    EXPECT_TRUE((test<float, float>(GetParam(), c, s)));
}
TEST_P(RotgTests, RealDoublePrecision) {
    double c(2.0);
    double s(-0.5);
    EXPECT_TRUE((test<double, double>(GetParam(), c, s)));
    EXPECT_TRUE((test<double, double>(GetParam(), c, s)));
    EXPECT_TRUE((test<double, double>(GetParam(), c, s)));
}
TEST_P(RotgTests, ComplexSinglePrecision) {
    float c = 2.0;
    float s = -0.5;
    EXPECT_TRUE((test<std::complex<float>, float>(GetParam(), c, s)));
    EXPECT_TRUE((test<std::complex<float>, float>(GetParam(), c, s)));
    EXPECT_TRUE((test<std::complex<float>, float>(GetParam(), c, s)));
}
TEST_P(RotgTests, ComplexDoublePrecision) {
    double c = 2.0;
    double s = -0.5;
    EXPECT_TRUE((test<std::complex<double>, double>(GetParam(), c, s)));
    EXPECT_TRUE((test<std::complex<double>, double>(GetParam(), c, s)));
    EXPECT_TRUE((test<std::complex<double>, double>(GetParam(), c, s)));
}

INSTANTIATE_TEST_SUITE_P(RotgTestSuite, RotgTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
