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
bool test(const device& dev, onemkl::uplo upper_lower, onemkl::transpose transa,
          onemkl::diag unit_nonunit, int n, int incx) {
    // Prepare data.
    vector<fp> x, x_ref, A;
    rand_vector(x, n, incx);
    x_ref = x;
    rand_trsm_matrix(A, transa, n, n, n);

    // Call Reference TPSV.
    const int n_ref = n, incx_ref = incx;
    using fp_ref = typename ref_type_info<fp>::type;

    ::tpsv(convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
           convert_to_cblas_diag(unit_nonunit), &n_ref, (fp_ref*)A.data(), (fp_ref*)x_ref.data(),
           &incx_ref);

    // Call DPC++ TPSV.

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

    buffer<fp, 1> x_buffer = make_buffer(x);
    buffer<fp, 1> A_buffer = make_buffer(A);

    try {
#ifdef CALL_RT_API
        onemkl::blas::tpsv(main_queue, upper_lower, transa, unit_nonunit, n, A_buffer, x_buffer,
                           incx);
#else
        TEST_RUN_CT(main_queue, onemkl::blas::tpsv,
                    (main_queue, upper_lower, transa, unit_nonunit, n, A_buffer, x_buffer, incx));
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during TPSV:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good;
    {
        auto x_accessor = x_buffer.template get_access<access::mode::read>();
        good            = check_equal_trsv_vector(x_accessor, x_ref, n, incx, n, std::cout);
    }

    return good;
}

class TpsvTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(TpsvTests, RealSinglePrecision) {
    EXPECT_TRUE(test<float>(GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans,
                            onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<float>(GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans,
                            onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<float>(GetParam(), onemkl::uplo::lower, onemkl::transpose::trans,
                            onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<float>(GetParam(), onemkl::uplo::upper, onemkl::transpose::trans,
                            onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<float>(GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans,
                            onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUE(test<float>(GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans,
                            onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUE(test<float>(GetParam(), onemkl::uplo::lower, onemkl::transpose::trans,
                            onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUE(test<float>(GetParam(), onemkl::uplo::upper, onemkl::transpose::trans,
                            onemkl::diag::nonunit, 30, 2));
}
TEST_P(TpsvTests, RealDoublePrecision) {
    EXPECT_TRUE(test<double>(GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans,
                             onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<double>(GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans,
                             onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<double>(GetParam(), onemkl::uplo::lower, onemkl::transpose::trans,
                             onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<double>(GetParam(), onemkl::uplo::upper, onemkl::transpose::trans,
                             onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<double>(GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans,
                             onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUE(test<double>(GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans,
                             onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUE(test<double>(GetParam(), onemkl::uplo::lower, onemkl::transpose::trans,
                             onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUE(test<double>(GetParam(), onemkl::uplo::upper, onemkl::transpose::trans,
                             onemkl::diag::nonunit, 30, 2));
}
TEST_P(TpsvTests, ComplexSinglePrecision) {
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::uplo::lower,
                                          onemkl::transpose::nontrans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::uplo::upper,
                                          onemkl::transpose::nontrans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::uplo::lower, onemkl::transpose::trans,
                                          onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::uplo::upper, onemkl::transpose::trans,
                                          onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::uplo::lower,
                                          onemkl::transpose::conjtrans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::uplo::upper,
                                          onemkl::transpose::conjtrans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::uplo::lower,
                                          onemkl::transpose::nontrans, onemkl::diag::nonunit, 30,
                                          2));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::uplo::upper,
                                          onemkl::transpose::nontrans, onemkl::diag::nonunit, 30,
                                          2));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::uplo::lower, onemkl::transpose::trans,
                                          onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::uplo::upper, onemkl::transpose::trans,
                                          onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::uplo::lower,
                                          onemkl::transpose::conjtrans, onemkl::diag::nonunit, 30,
                                          2));
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), onemkl::uplo::upper,
                                          onemkl::transpose::conjtrans, onemkl::diag::nonunit, 30,
                                          2));
}
TEST_P(TpsvTests, ComplexDoublePrecision) {
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::uplo::lower,
                                           onemkl::transpose::nontrans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::uplo::upper,
                                           onemkl::transpose::nontrans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::uplo::lower,
                                           onemkl::transpose::trans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::uplo::upper,
                                           onemkl::transpose::trans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<std::complex<double>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::conjtrans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<std::complex<double>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::conjtrans, onemkl::diag::unit, 30, 2));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::uplo::lower,
                                           onemkl::transpose::nontrans, onemkl::diag::nonunit, 30,
                                           2));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::uplo::upper,
                                           onemkl::transpose::nontrans, onemkl::diag::nonunit, 30,
                                           2));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::uplo::lower,
                                           onemkl::transpose::trans, onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::uplo::upper,
                                           onemkl::transpose::trans, onemkl::diag::nonunit, 30, 2));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::uplo::lower,
                                           onemkl::transpose::conjtrans, onemkl::diag::nonunit, 30,
                                           2));
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), onemkl::uplo::upper,
                                           onemkl::transpose::conjtrans, onemkl::diag::nonunit, 30,
                                           2));
}

INSTANTIATE_TEST_SUITE_P(TpsvTestSuite, TpsvTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
