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

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include <CL/sycl.hpp>
#include "cblas.h"
#include "oneapi/mkl.hpp"
#include "oneapi/mkl/detail/config.hpp"
#include "onemkl_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

using namespace cl::sycl;
using std::vector;

extern std::vector<cl::sycl::device *> devices;

namespace {

template <typename fp>
int test(device *dev, oneapi::mkl::layout layout, oneapi::mkl::uplo upper_lower, int n, int k,
         fp alpha, fp beta, int incx, int incy, int lda) {
    // Prepare data.
    vector<fp> x, y, y_ref, A;

    rand_vector(x, n, incx);
    rand_vector(y, n, incy);
    y_ref = y;
    rand_matrix(A, layout, oneapi::mkl::transpose::nontrans, n, n, lda);

    // Call Reference HBMV.
    const int n_ref = n, incx_ref = incx, incy_ref = incy, lda_ref = lda;
    const int k_ref = k;
    using fp_ref = typename ref_type_info<fp>::type;

    ::hbmv(convert_to_cblas_layout(layout), convert_to_cblas_uplo(upper_lower), &n_ref, &k_ref,
           (fp_ref *)&alpha, (fp_ref *)A.data(), &lda_ref, (fp_ref *)x.data(), &incx_ref,
           (fp_ref *)&beta, (fp_ref *)y_ref.data(), &incy_ref);

    // Call DPC++ HBMV.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during HBMV:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<fp, 1> x_buffer = make_buffer(x);
    buffer<fp, 1> y_buffer = make_buffer(y);
    buffer<fp, 1> A_buffer = make_buffer(A);

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                oneapi::mkl::blas::column_major::hbmv(main_queue, upper_lower, n, k, alpha,
                                                      A_buffer, lda, x_buffer, incx, beta, y_buffer,
                                                      incy);
                break;
            case oneapi::mkl::layout::row_major:
                oneapi::mkl::blas::row_major::hbmv(main_queue, upper_lower, n, k, alpha, A_buffer,
                                                   lda, x_buffer, incx, beta, y_buffer, incy);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::hbmv, upper_lower,
                                   n, k, alpha, A_buffer, lda, x_buffer, incx, beta, y_buffer,
                                   incy);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::hbmv, upper_lower, n,
                                   k, alpha, A_buffer, lda, x_buffer, incx, beta, y_buffer, incy);
                break;
            default: break;
        }
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during HBMV:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const oneapi::mkl::unimplemented &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of HBMV:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    auto y_accessor = y_buffer.template get_access<access::mode::read>();
    bool good = check_equal_vector(y_accessor, y_ref, n, incy, n, std::cout);

    return (int)good;
}

class HbmvTests
        : public ::testing::TestWithParam<std::tuple<cl::sycl::device *, oneapi::mkl::layout>> {};

TEST_P(HbmvTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0, -0.5);
    std::complex<float> beta(3.0, -1.5);
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::mkl::uplo::lower, 30, 5, alpha, beta, 2, 3,
                                                42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::mkl::uplo::upper, 30, 5, alpha, beta, 2, 3,
                                                42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::mkl::uplo::lower, 30, 5, alpha, beta, -2,
                                                -3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::mkl::uplo::upper, 30, 5, alpha, beta, -2,
                                                -3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::mkl::uplo::lower, 30, 5, alpha, beta, 1, 1,
                                                42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::mkl::uplo::upper, 30, 5, alpha, beta, 1, 1,
                                                42));
}
TEST_P(HbmvTests, ComplexDoublePrecision) {
    std::complex<double> alpha(2.0, -0.5);
    std::complex<double> beta(3.0, -1.5);
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::mkl::uplo::lower, 30, 5, alpha, beta, 2, 3,
                                                 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::mkl::uplo::upper, 30, 5, alpha, beta, 2, 3,
                                                 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::mkl::uplo::lower, 30, 5, alpha, beta, -2,
                                                 -3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::mkl::uplo::upper, 30, 5, alpha, beta, -2,
                                                 -3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::mkl::uplo::lower, 30, 5, alpha, beta, 1, 1,
                                                 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::mkl::uplo::upper, 30, 5, alpha, beta, 1, 1,
                                                 42));
}

INSTANTIATE_TEST_SUITE_P(HbmvTestSuite, HbmvTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::column_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
