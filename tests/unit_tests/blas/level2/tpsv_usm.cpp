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
int test(const device& dev, onemkl::uplo upper_lower, onemkl::transpose transa,
         onemkl::diag unit_nonunit, int n, int incx) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during TPSV:\n"
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
    vector<fp, decltype(ua)> x(ua), A(ua);
    rand_vector(x, n, incx);
    rand_trsm_matrix(A, transa, n, n, n);

    auto x_ref = x;

    // Call Reference TPSV.
    const int n_ref = n, incx_ref = incx;
    using fp_ref = typename ref_type_info<fp>::type;

    ::tpsv(convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
           convert_to_cblas_diag(unit_nonunit), &n_ref, (fp_ref*)A.data(), (fp_ref*)x_ref.data(),
           &incx_ref);

    // Call DPC++ TPSV.

    try {
#ifdef CALL_RT_API
        done = onemkl::blas::tpsv(main_queue, upper_lower, transa, unit_nonunit, n, A.data(),
                                  x.data(), incx, dependencies);
        done.wait();
#else
        TEST_RUN_CT(main_queue, onemkl::blas::tpsv,
                    (main_queue, upper_lower, transa, unit_nonunit, n, A.data(), x.data(), incx,
                     dependencies));
    #ifndef ENABLE_CUBLAS_BACKEND
        main_queue.wait();
    #endif
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during TPSV:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const onemkl::backend_unsupported_exception& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of TPSV:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good = check_equal_trsv_vector(x, x_ref, n, incx, n, std::cout);

    return (int)good;
}

class TpsvUsmTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(TpsvUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans,
                                  onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans,
                                  onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::lower, onemkl::transpose::trans,
                                  onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::upper, onemkl::transpose::trans,
                                  onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans,
                                  onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans,
                                  onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::lower, onemkl::transpose::trans,
                                  onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::upper, onemkl::transpose::trans,
                                  onemkl::diag::nonunit, 30, 2));
}
TEST_P(TpsvUsmTests, RealDoublePrecision) {
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans,
                                   onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans,
                                   onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::lower, onemkl::transpose::trans,
                                   onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::upper, onemkl::transpose::trans,
                                   onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans,
                                   onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans,
                                   onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::lower, onemkl::transpose::trans,
                                   onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::upper, onemkl::transpose::trans,
                                   onemkl::diag::nonunit, 30, 2));
}
TEST_P(TpsvUsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::trans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::trans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::conjtrans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::conjtrans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::uplo::lower,
                                                onemkl::transpose::nontrans, onemkl::diag::nonunit,
                                                30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::uplo::upper,
                                                onemkl::transpose::nontrans, onemkl::diag::nonunit,
                                                30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::trans, onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::trans, onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::uplo::lower,
                                                onemkl::transpose::conjtrans, onemkl::diag::nonunit,
                                                30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::uplo::upper,
                                                onemkl::transpose::conjtrans, onemkl::diag::nonunit,
                                                30, 2));
}
TEST_P(TpsvUsmTests, ComplexDoublePrecision) {
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::trans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::trans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::conjtrans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::conjtrans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::uplo::lower,
                                                 onemkl::transpose::nontrans, onemkl::diag::nonunit,
                                                 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::uplo::upper,
                                                 onemkl::transpose::nontrans, onemkl::diag::nonunit,
                                                 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::trans, onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::trans, onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::uplo::lower,
                                                 onemkl::transpose::conjtrans,
                                                 onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::uplo::upper,
                                                 onemkl::transpose::conjtrans,
                                                 onemkl::diag::nonunit, 30, 2));
}

INSTANTIATE_TEST_SUITE_P(TpsvUsmTestSuite, TpsvUsmTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
