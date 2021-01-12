/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

extern std::vector<cl::sycl::device *> devices;

namespace {

template <typename fp, typename fp_scalar>
int test(device *dev, oneapi::mkl::layout layout, int N, int incx, int incy, fp_scalar c,
         fp_scalar s) {
    // Prepare data.
    vector<fp> x, x_ref, y, y_ref;
    rand_vector(x, N, incx);
    rand_vector(y, N, incy);
    y_ref = y;
    x_ref = x;

    // Call Reference ROT.
    using fp_ref = typename ref_type_info<fp>::type;
    const int N_ref = N, incx_ref = incx, incy_ref = incy;

    ::rot(&N_ref, (fp_ref *)x_ref.data(), &incx_ref, (fp_ref *)y_ref.data(), &incy_ref,
          (fp_scalar *)&c, (fp_scalar *)&s);

    // Call DPC++ ROT.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during ROT:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<fp, 1> x_buffer = make_buffer(x);
    buffer<fp, 1> y_buffer = make_buffer(y);

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                oneapi::mkl::blas::column_major::rot(main_queue, N, x_buffer, incx, y_buffer, incy,
                                                     c, s);
                break;
            case oneapi::mkl::layout::row_major:
                oneapi::mkl::blas::row_major::rot(main_queue, N, x_buffer, incx, y_buffer, incy, c,
                                                  s);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::rot, N, x_buffer,
                                   incx, y_buffer, incy, c, s);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::rot, N, x_buffer, incx,
                                   y_buffer, incy, c, s);
                break;
            default: break;
        }
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during ROT:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const oneapi::mkl::unimplemented &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of ROT:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    auto x_accessor = x_buffer.template get_access<access::mode::read>();
    bool good_x = check_equal_vector(x_accessor, x_ref, N, incx, N, std::cout);
    auto y_accessor = y_buffer.template get_access<access::mode::read>();
    bool good_y = check_equal_vector(y_accessor, y_ref, N, incy, N, std::cout);

    bool good = good_x && good_y;

    return (int)good;
}

class RotTests
        : public ::testing::TestWithParam<std::tuple<cl::sycl::device *, oneapi::mkl::layout>> {};

TEST_P(RotTests, RealSinglePrecision) {
    float c(2.0);
    float s(-0.5);
    EXPECT_TRUEORSKIP(
        (test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, 2, 3, c, s)));
    EXPECT_TRUEORSKIP(
        (test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, 1, 1, c, s)));
    EXPECT_TRUEORSKIP(
        (test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, -2, -3, c, s)));
}
TEST_P(RotTests, RealDoublePrecision) {
    double c(2.0);
    double s(-0.5);
    EXPECT_TRUEORSKIP(
        (test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, 2, 3, c, s)));
    EXPECT_TRUEORSKIP(
        (test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, 1, 1, c, s)));
    EXPECT_TRUEORSKIP((test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1357,
                                            -2, -3, c, s)));
}
TEST_P(RotTests, ComplexSinglePrecision) {
    float c = 2.0;
    float s = -0.5;
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, 2, 3, c, s)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, 1, 1, c, s)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, -2, -3, c, s)));
}
TEST_P(RotTests, ComplexDoublePrecision) {
    double c = 2.0;
    double s = -0.5;
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, 2, 3, c, s)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, 1, 1, c, s)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, -2, -3, c, s)));
}

INSTANTIATE_TEST_SUITE_P(RotTestSuite, RotTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::column_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
