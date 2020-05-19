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
#include "usm_allocator_helper.hpp"

#include <gtest/gtest.h>

using namespace cl::sycl;
using std::vector;

extern std::vector<cl::sycl::device> devices;

namespace {

template <typename fp>
int test(const device &dev, onemkl::transpose transa, int m, int n, int kl, int ku, fp alpha,
         fp beta, int incx, int incy, int lda) {
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
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.
    auto ua = usm_allocator_helper<fp, 64>(cxt, dev);
    vector<fp, decltype(ua)> x(ua), y(ua), y_ref(ua), A(ua);
    int x_len = outer_dimension(transa, m, n);
    int y_len = inner_dimension(transa, m, n);

    rand_vector(x, x_len, incx);
    rand_vector(y, y_len, incy);
    rand_matrix(A, onemkl::transpose::nontrans, m, n, lda);

    y_ref.resize(y.size());
    for (int i = 0; i < y.size(); i++)
        y_ref[i] = y[i];

    // Call Reference GBMV.
    const int m_ref = m, n_ref = n, incx_ref = incx, incy_ref = incy, lda_ref = lda;
    int kl_ref = kl, ku_ref = ku;
    using fp_ref = typename ref_type_info<fp>::type;

    ::gbmv(convert_to_cblas_trans(transa), &m_ref, &n_ref, &kl_ref, &ku_ref, (fp_ref *)&alpha,
           (fp_ref *)A.data(), &lda_ref, (fp_ref *)x.data(), &incx_ref, (fp_ref *)&beta,
           (fp_ref *)y_ref.data(), &incy_ref);

    // Call DPC++ GBMV.

    try {
#ifdef CALL_RT_API
        done = onemkl::blas::gbmv(main_queue, transa, m, n, kl, ku, alpha, A.data(), lda, x.data(),
                                  incx, beta, y.data(), incy, dependencies);
        done.wait();
#else
        TEST_RUN_CT(main_queue, onemkl::blas::gbmv,
                    (main_queue, transa, m, n, kl, ku, alpha, A.data(), lda, x.data(), incx, beta,
                     y.data(), incy, dependencies));
        main_queue.wait();
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during GBMV:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const backend_unsupported_exception &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of GBMV_USM:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good;
    { good = check_equal_vector(y, y_ref, y_len, incy, std::max<int>(m, n), std::cout); }

    return (int)good;
}

class GbmvUsmTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(GbmvUsmTests, RealSinglePrecision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUEORSKIP(
        test<float>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5, 7, alpha, beta, 2, 3, 42));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5, 7, alpha,
                                  beta, -2, -3, 42));
    EXPECT_TRUEORSKIP(
        test<float>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5, 7, alpha, beta, 1, 1, 42));
    EXPECT_TRUEORSKIP(
        test<float>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7, alpha, beta, 2, 3, 42));
    EXPECT_TRUEORSKIP(
        test<float>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7, alpha, beta, -2, -3, 42));
    EXPECT_TRUEORSKIP(
        test<float>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7, alpha, beta, 1, 1, 42));
}
TEST_P(GbmvUsmTests, RealDoublePrecision) {
    double alpha(2.0);
    double beta(3.0);
    EXPECT_TRUEORSKIP(
        test<double>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5, 7, alpha, beta, 2, 3, 42));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5, 7, alpha,
                                   beta, -2, -3, 42));
    EXPECT_TRUEORSKIP(
        test<double>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5, 7, alpha, beta, 1, 1, 42));
    EXPECT_TRUEORSKIP(
        test<double>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7, alpha, beta, 2, 3, 42));
    EXPECT_TRUEORSKIP(
        test<double>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7, alpha, beta, -2, -3, 42));
    EXPECT_TRUEORSKIP(
        test<double>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7, alpha, beta, 1, 1, 42));
}
TEST_P(GbmvUsmTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0, -0.5);
    std::complex<float> beta(3.0, -1.5);
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5,
                                                7, alpha, beta, 2, 3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5,
                                                7, alpha, beta, -2, -3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5,
                                                7, alpha, beta, 1, 1, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7,
                                                alpha, beta, 2, 3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7,
                                                alpha, beta, -2, -3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7,
                                                alpha, beta, 1, 1, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::transpose::conjtrans, 25, 30, 5,
                                                7, alpha, beta, 2, 3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::transpose::conjtrans, 25, 30, 5,
                                                7, alpha, beta, -2, -3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::transpose::conjtrans, 25, 30, 5,
                                                7, alpha, beta, 1, 1, 42));
}
TEST_P(GbmvUsmTests, ComplexDoublePrecision) {
    std::complex<double> alpha(2.0, -0.5);
    std::complex<double> beta(3.0, -1.5);
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5,
                                                 7, alpha, beta, 2, 3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5,
                                                 7, alpha, beta, -2, -3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::transpose::nontrans, 25, 30, 5,
                                                 7, alpha, beta, 1, 1, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7,
                                                 alpha, beta, 2, 3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7,
                                                 alpha, beta, -2, -3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::transpose::trans, 25, 30, 5, 7,
                                                 alpha, beta, 1, 1, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::transpose::conjtrans, 25, 30,
                                                 5, 7, alpha, beta, 2, 3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::transpose::conjtrans, 25, 30,
                                                 5, 7, alpha, beta, -2, -3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::transpose::conjtrans, 25, 30,
                                                 5, 7, alpha, beta, 1, 1, 42));
}

INSTANTIATE_TEST_SUITE_P(GbmvUsmTestSuite, GbmvUsmTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
