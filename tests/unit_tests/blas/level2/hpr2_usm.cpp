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
int test(const device &dev, onemkl::uplo upper_lower, int n, fp alpha, int incx, int incy) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during HPR2:\n"
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
    auto ua = usm_allocator<fp, usm::alloc::shared, 64>(cxt, dev);
    vector<fp, decltype(ua)> x(ua), y(ua), A(ua);
    rand_vector(x, n, incx);
    rand_vector(y, n, incy);
    rand_matrix(A, onemkl::transpose::nontrans, n, n, n);

    auto A_ref = A;

    // Call Reference HPR2.
    const int n_ref = n, incx_ref = incx, incy_ref = incy;
    using fp_ref = typename ref_type_info<fp>::type;

    ::hpr2(convert_to_cblas_uplo(upper_lower), &n_ref, (fp_ref *)&alpha, (fp_ref *)x.data(),
           &incx_ref, (fp_ref *)y.data(), &incy_ref, (fp_ref *)A_ref.data());

    // Call DPC++ HPR2.

    try {
#ifdef CALL_RT_API
        done = onemkl::blas::hpr2(main_queue, upper_lower, n, alpha, x.data(), incx, y.data(), incy,
                                  A.data(), dependencies);
        done.wait();
#else
        TEST_RUN_CT(main_queue, onemkl::blas::hpr2,
                    (main_queue, upper_lower, n, alpha, x.data(), incx, y.data(), incy, A.data(),
                     dependencies));
    #ifndef ENABLE_CUBLAS_BACKEND
        main_queue.wait();
    #endif
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during HPR2:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const onemkl::backend_unsupported_exception &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of HPR2:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good = check_equal_matrix(A, A_ref, n, n, n, n, std::cout);

    return (int)good;
}

class Hpr2UsmTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(Hpr2UsmTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0, -0.5);
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::uplo::lower, 30, alpha, 2, 3));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::uplo::upper, 30, alpha, 2, 3));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), onemkl::uplo::lower, 30, alpha, -2, -3));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), onemkl::uplo::upper, 30, alpha, -2, -3));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::uplo::lower, 30, alpha, 1, 1));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::uplo::upper, 30, alpha, 1, 1));
}
TEST_P(Hpr2UsmTests, ComplexDoublePrecision) {
    std::complex<double> alpha(2.0, -0.5);
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::uplo::lower, 30, alpha, 2, 3));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::uplo::upper, 30, alpha, 2, 3));
    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(GetParam(), onemkl::uplo::lower, 30, alpha, -2, -3));
    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(GetParam(), onemkl::uplo::upper, 30, alpha, -2, -3));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::uplo::lower, 30, alpha, 1, 1));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::uplo::upper, 30, alpha, 1, 1));
}

INSTANTIATE_TEST_SUITE_P(Hpr2UsmTestSuite, Hpr2UsmTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
