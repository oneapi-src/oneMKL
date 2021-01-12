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

extern std::vector<cl::sycl::device*> devices;

namespace {

template <typename fp, typename fp_scalar>
int test(device* dev, oneapi::mkl::layout layout, int N, int incx, fp_scalar alpha) {
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

    queue main_queue(*dev, exception_handler);
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.
    auto ua = usm_allocator<fp, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp, decltype(ua)> x(ua);

    rand_vector(x, N, incx);

    auto x_ref = x;

    // Call Reference SCAL.
    using fp_ref = typename ref_type_info<fp>::type;
    using fp_scalar_mkl = typename ref_type_info<fp_scalar>::type;

    const int N_ref = N, incx_ref = std::abs(incx);

    ::scal(&N_ref, (fp_scalar_mkl*)&alpha, (fp_ref*)x_ref.data(), &incx_ref);

    // Call DPC++ SCAL.

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                done = oneapi::mkl::blas::column_major::scal(main_queue, N, alpha, x.data(), incx,
                                                             dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                done = oneapi::mkl::blas::row_major::scal(main_queue, N, alpha, x.data(), incx,
                                                          dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::scal, N, alpha,
                                   x.data(), incx, dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::scal, N, alpha,
                                   x.data(), incx, dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during SCAL:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const oneapi::mkl::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of SCAL:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good = check_equal_vector(x, x_ref, N, incx, N, std::cout);

    return (int)good;
}

class ScalUsmTests
        : public ::testing::TestWithParam<std::tuple<cl::sycl::device*, oneapi::mkl::layout>> {};

TEST_P(ScalUsmTests, RealSinglePrecision) {
    float alpha(2.0);
    EXPECT_TRUEORSKIP(
        (test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, 2, alpha)));
    EXPECT_TRUEORSKIP(
        (test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, -3, alpha)));
}
TEST_P(ScalUsmTests, RealDoublePrecision) {
    double alpha(2.0);
    EXPECT_TRUEORSKIP(
        (test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, 2, alpha)));
    EXPECT_TRUEORSKIP(
        (test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, -3, alpha)));
}
TEST_P(ScalUsmTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0, -0.5);
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, 2, alpha)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, -3, alpha)));
}
TEST_P(ScalUsmTests, ComplexDoublePrecision) {
    std::complex<double> alpha(2.0, -0.5);
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, 2, alpha)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, -3, alpha)));
}
TEST_P(ScalUsmTests, ComplexRealSinglePrecision) {
    float alpha(2.0);
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(std::get<0>(GetParam()),
                                                        std::get<1>(GetParam()), 1357, 2, alpha)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(std::get<0>(GetParam()),
                                                        std::get<1>(GetParam()), 1357, -3, alpha)));
}
TEST_P(ScalUsmTests, ComplexRealDoublePrecision) {
    double alpha(2.0);
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, 2, alpha)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, -3, alpha)));
}

INSTANTIATE_TEST_SUITE_P(ScalUsmTestSuite, ScalUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::column_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
