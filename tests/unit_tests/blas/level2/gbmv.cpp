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
#include "config.hpp"
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
bool test(const device &dev, onemkl::transpose transa, int m, int n, int kl, int ku, fp alpha,
          fp beta, int incx, int incy, int lda) {
    // Prepare data.
    int x_len = outer_dimension(transa, m, n);
    int y_len = inner_dimension(transa, m, n);

    vector<fp> x, y, y_ref, A;

    rand_vector(x, x_len, incx);
    rand_vector(y, y_len, incy);
    y_ref = y;
    rand_matrix(A, onemkl::transpose::nontrans, m, n, lda);

    // Call Reference GBMV.
    const int m_ref = m, n_ref = n, incx_ref = incx, incy_ref = incy, lda_ref = lda;
    int kl_ref = kl, ku_ref = ku;
    using fp_ref = typename ref_type_info<fp>::type;

    ::gbmv(convert_to_cblas_trans(transa), &m_ref, &n_ref, &kl_ref, &ku_ref, (fp_ref *)&alpha,
           (fp_ref *)A.data(), &lda_ref, (fp_ref *)x.data(), &incx_ref, (fp_ref *)&beta,
           (fp_ref *)y_ref.data(), &incy_ref);

    // Call DPC++ GBMV.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during GBMV:\n"
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
        onemkl::blas::gbmv(main_queue, transa, m, n, kl, ku, alpha, A_buffer, lda, x_buffer, incx,
                           beta, y_buffer, incy);
#else
        TEST_RUN_CT(main_queue, onemkl::blas::gbmv,
                    (main_queue, transa, m, n, kl, ku, alpha, A_buffer, lda, x_buffer, incx, beta,
                     y_buffer, incy));
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during GBMV:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good;
    {
        auto y_accessor = y_buffer.template get_access<access::mode::read>();
        good = check_equal_vector(y_accessor, y_ref, y_len, incy, std::max<int>(m, n), std::cout);
    }

    return good;
}

class GbmvTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(GbmvTests, RealSinglePrecision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUE(
        test<float>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5, 7, alpha, beta, 2, 3, 42));
    EXPECT_TRUE(test<float>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5, 7, alpha, beta, -2,
                            -3, 42));
    EXPECT_TRUE(
        test<float>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5, 7, alpha, beta, 1, 1, 42));
    EXPECT_TRUE(
        test<float>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7, alpha, beta, 2, 3, 42));
    EXPECT_TRUE(
        test<float>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7, alpha, beta, -2, -3, 42));
    EXPECT_TRUE(
        test<float>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7, alpha, beta, 1, 1, 42));
}
TEST_P(GbmvTests, RealDoublePrecision) {
    double alpha(2.0);
    double beta(3.0);
    EXPECT_TRUE(
        test<double>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5, 7, alpha, beta, 2, 3, 42));
    EXPECT_TRUE(test<double>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5, 7, alpha, beta, -2,
                             -3, 42));
    EXPECT_TRUE(
        test<double>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5, 7, alpha, beta, 1, 1, 42));
    EXPECT_TRUE(
        test<double>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7, alpha, beta, 2, 3, 42));
    EXPECT_TRUE(
        test<double>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7, alpha, beta, -2, -3, 42));
    EXPECT_TRUE(
        test<double>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7, alpha, beta, 1, 1, 42));
}
TEST_P(GbmvTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0, -0.5);
    std::complex<float> beta(3.0, -1.5);
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5, 7,
                                          alpha, beta, 2, 3, 42));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5, 7,
                                          alpha, beta, -2, -3, 42));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5, 7,
                                          alpha, beta, 1, 1, 42));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7, alpha,
                                          beta, 2, 3, 42));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7, alpha,
                                          beta, -2, -3, 42));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7, alpha,
                                          beta, 1, 1, 42));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::transpose::conjtrans, 25, 30, 5, 7,
                                          alpha, beta, 2, 3, 42));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::transpose::conjtrans, 25, 30, 5, 7,
                                          alpha, beta, -2, -3, 42));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::transpose::conjtrans, 25, 30, 5, 7,
                                          alpha, beta, 1, 1, 42));
}
TEST_P(GbmvTests, ComplexDoublePrecision) {
    std::complex<double> alpha(2.0, -0.5);
    std::complex<double> beta(3.0, -1.5);
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5, 7,
                                           alpha, beta, 2, 3, 42));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5, 7,
                                           alpha, beta, -2, -3, 42));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5, 7,
                                           alpha, beta, 1, 1, 42));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7,
                                           alpha, beta, 2, 3, 42));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7,
                                           alpha, beta, -2, -3, 42));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7,
                                           alpha, beta, 1, 1, 42));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::transpose::conjtrans, 25, 30, 5, 7,
                                           alpha, beta, 2, 3, 42));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::transpose::conjtrans, 25, 30, 5, 7,
                                           alpha, beta, -2, -3, 42));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::transpose::conjtrans, 25, 30, 5, 7,
                                           alpha, beta, 1, 1, 42));
}

INSTANTIATE_TEST_SUITE_P(GbmvTestSuite, GbmvTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
