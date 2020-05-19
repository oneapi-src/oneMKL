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
int test(const device& dev) {
    // Prepare data.
    fp d1, d2, x1, y1, d1_ref, d2_ref, x1_ref;
    vector<fp> param(5, fp(0)), param_ref(5, fp(0));

    d1     = rand_scalar<fp>();
    d1     = abs(d1);
    d2     = rand_scalar<fp>();
    x1     = rand_scalar<fp>();
    y1     = rand_scalar<fp>();
    d1_ref = d1;
    d2_ref = d2;
    x1_ref = x1;

    // Call Reference ROTMG.

    ::rotmg(&d1_ref, &d2_ref, &x1_ref, &y1, (fp*)param_ref.data());

    // Call DPC++ ROTMG.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during ROTMG:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(dev, exception_handler);

    buffer<fp, 1> d1_buffer(&d1, range<1>(1));
    buffer<fp, 1> d2_buffer(&d2, range<1>(1));
    buffer<fp, 1> x1_buffer(&x1, range<1>(1));
    buffer<fp, 1> param_buffer = make_buffer(param);
    try {
#ifdef CALL_RT_API
        onemkl::blas::rotmg(main_queue, d1_buffer, d2_buffer, x1_buffer, y1, param_buffer);
#else
        TEST_RUN_CT(main_queue, onemkl::blas::rotmg,
                    (main_queue, d1_buffer, d2_buffer, x1_buffer, y1, param_buffer));
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during ROTMG:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const onemkl::backend_unsupported_exception& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of ROTMG:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good;
    {
        auto d1_accessor    = d1_buffer.template get_access<access::mode::read>();
        bool good_d1        = check_equal(d1_accessor[0], d1_ref, 1, std::cout);
        auto d2_accessor    = d2_buffer.template get_access<access::mode::read>();
        bool good_d2        = check_equal(d2_accessor[0], d2_ref, 1, std::cout);
        auto x1_accessor    = x1_buffer.template get_access<access::mode::read>();
        bool good_x1        = check_equal(x1_accessor[0], x1_ref, 1, std::cout);
        auto param_accessor = param_buffer.template get_access<access::mode::read>();
        bool good_param     = check_equal_vector(param_accessor, param_ref, 5, 1, 1, std::cout);
        good                = good_d1 && good_d2 && good_x1 && good_param;
    }

    return (int)good;
}

class RotmgTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(RotmgTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(GetParam()));
}
TEST_P(RotmgTests, RealDoublePrecision) {
    EXPECT_TRUEORSKIP(test<double>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(RotmgTestSuite, RotmgTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
