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

int test(device *dev, oneapi::mkl::layout layout, int N, int incx, int incy, float alpha) {
    // Prepare data.
    vector<float> x, y;
    float result = float(-1), result_ref = float(-1);

    rand_vector(x, N, incx);
    rand_vector(y, N, incy);

    // Call Reference SDSDOT.
    const int N_ref = N, incx_ref = incx, incy_ref = incy;

    result_ref = ::sdsdot(&N_ref, (float *)&alpha, (float *)x.data(), &incx_ref, (float *)y.data(),
                          &incy_ref);

    // Call DPC++ SDSDOT.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during SDSDOT:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<float, 1> x_buffer = make_buffer(x);
    buffer<float, 1> y_buffer = make_buffer(y);
    buffer<float, 1> result_buffer(&result, range<1>(1));

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                oneapi::mkl::blas::column_major::sdsdot(main_queue, N, alpha, x_buffer, incx,
                                                        y_buffer, incy, result_buffer);
                break;
            case oneapi::mkl::layout::row_major:
                oneapi::mkl::blas::row_major::sdsdot(main_queue, N, alpha, x_buffer, incx, y_buffer,
                                                     incy, result_buffer);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::sdsdot, N, alpha,
                                   x_buffer, incx, y_buffer, incy, result_buffer);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::sdsdot, N, alpha,
                                   x_buffer, incx, y_buffer, incy, result_buffer);
                break;
            default: break;
        }
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during SDSDOT:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const oneapi::mkl::unimplemented &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of SDSDOT:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    auto result_accessor = result_buffer.template get_access<access::mode::read>();
    bool good = check_equal(result_accessor[0], result_ref, N, std::cout);

    return (int)good;
}

class SdsdotTests
        : public ::testing::TestWithParam<std::tuple<cl::sycl::device *, oneapi::mkl::layout>> {};

TEST_P(SdsdotTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test(std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, 2, 3, 2.0));
    EXPECT_TRUEORSKIP(test(std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, -2, -3, 2.0));
    EXPECT_TRUEORSKIP(test(std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, 1, 1, 2.0));
}

INSTANTIATE_TEST_SUITE_P(SdsdotTestSuite, SdsdotTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::column_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
