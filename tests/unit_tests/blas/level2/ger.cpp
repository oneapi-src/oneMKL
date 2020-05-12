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
bool test(const device &dev, int m, int n, fp alpha, int incx, int incy, int lda) {
    // Prepare data.

    vector<fp> x, y, A_ref, A;

    rand_vector(x, m, incx);
    rand_vector(y, n, incy);
    rand_matrix(A, onemkl::transpose::nontrans, m, n, lda);
    A_ref = A;

    // Call Reference GER.
    const int m_ref = m, n_ref = n, incx_ref = incx, incy_ref = incy, lda_ref = lda;
    using fp_ref = typename ref_type_info<fp>::type;

    ::ger(&m_ref, &n_ref, (fp_ref *)&alpha, (fp_ref *)x.data(), &incx_ref, (fp_ref *)y.data(),
          &incy_ref, (fp_ref *)A_ref.data(), &lda_ref);

    // Call DPC++ GER.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during GER:\n"
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
        onemkl::blas::ger(main_queue, m, n, alpha, x_buffer, incx, y_buffer, incy, A_buffer, lda);
#else
        TEST_RUN_CT(main_queue, onemkl::blas::ger,
                    (main_queue, m, n, alpha, x_buffer, incx, y_buffer, incy, A_buffer, lda));
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during GER:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good;
    {
        auto A_accessor = A_buffer.template get_access<access::mode::read>();
        good = check_equal_matrix(A_accessor, A_ref, m, n, lda, std::max<int>(m, n), std::cout);
    }

    return good;
}

class GerTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(GerTests, RealSinglePrecision) {
    float alpha(2.0);
    EXPECT_TRUE(test<float>(GetParam(), 25, 30, alpha, 2, 3, 42));
    EXPECT_TRUE(test<float>(GetParam(), 25, 30, alpha, -2, -3, 42));
    EXPECT_TRUE(test<float>(GetParam(), 25, 30, alpha, 1, 1, 42));
}
TEST_P(GerTests, RealDoublePrecision) {
    double alpha(2.0);
    EXPECT_TRUE(test<double>(GetParam(), 25, 30, alpha, 2, 3, 42));
    EXPECT_TRUE(test<double>(GetParam(), 25, 30, alpha, -2, -3, 42));
    EXPECT_TRUE(test<double>(GetParam(), 25, 30, alpha, 1, 1, 42));
}

INSTANTIATE_TEST_SUITE_P(GerTestSuite, GerTests, ::testing::ValuesIn(devices), ::DeviceNamePrint());

} // anonymous namespace
