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
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include <CL/sycl.hpp>
#include "allocator_helper.hpp"
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

extern std::vector<cl::sycl::device*> devices;

namespace {

template <typename fp, typename fp_scalar>
int test(device* dev, oneapi::mkl::layout layout, oneapi::mkl::uplo upper_lower,
         oneapi::mkl::transpose trans, int n, int k, int lda, int ldc, fp_scalar alpha,
         fp_scalar beta) {
    // Prepare data.
    vector<fp, allocator_helper<fp, 64>> A, C, C_ref;
    rand_matrix(A, layout, trans, n, k, lda);
    rand_matrix(C, layout, oneapi::mkl::transpose::nontrans, n, n, ldc);
    C_ref = C;

    // Call Reference HERK.
    const int n_ref = n, k_ref = k;
    const int lda_ref = lda, ldc_ref = ldc;

    using fp_ref = typename ref_type_info<fp>::type;

    ::herk(convert_to_cblas_layout(layout), convert_to_cblas_uplo(upper_lower),
           convert_to_cblas_trans(trans), &n_ref, &k_ref, (fp_scalar*)&alpha, (fp_ref*)A.data(),
           &lda_ref, (fp_scalar*)&beta, (fp_ref*)C_ref.data(), &ldc_ref);

    // Call DPC++ HERK.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during HERK:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.what() << std::endl;
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<fp, 1> A_buffer(A.data(), range<1>(A.size()));
    buffer<fp, 1> C_buffer(C.data(), range<1>(C.size()));

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                oneapi::mkl::blas::column_major::herk(main_queue, upper_lower, trans, n, k, alpha,
                                                      A_buffer, lda, beta, C_buffer, ldc);
                break;
            case oneapi::mkl::layout::row_major:
                oneapi::mkl::blas::row_major::herk(main_queue, upper_lower, trans, n, k, alpha,
                                                   A_buffer, lda, beta, C_buffer, ldc);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::herk, upper_lower,
                                   trans, n, k, alpha, A_buffer, lda, beta, C_buffer, ldc);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::herk, upper_lower,
                                   trans, n, k, alpha, A_buffer, lda, beta, C_buffer, ldc);
                break;
            default: break;
        }
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during HERK:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.what() << std::endl;
    }

    catch (const oneapi::mkl::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of HERK:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    auto C_accessor = C_buffer.template get_access<access::mode::read>();
    bool good =
        check_equal_matrix(C_accessor, C_ref, layout, n, n, ldc, 10 * std::max(n, k), std::cout);

    return (int)good;
}

class HerkTests
        : public ::testing::TestWithParam<std::tuple<cl::sycl::device*, oneapi::mkl::layout>> {};

TEST_P(HerkTests, ComplexSinglePrecision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::lower,
        oneapi::mkl::transpose::nontrans, 72, 27, 101, 103, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::nontrans, 72, 27, 101, 103, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::lower,
        oneapi::mkl::transpose::conjtrans, 72, 27, 101, 103, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::conjtrans, 72, 27, 101, 103, alpha, beta)));
}
TEST_P(HerkTests, ComplexDoublePrecision) {
    double alpha(2.0);
    double beta(3.0);
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::lower,
        oneapi::mkl::transpose::nontrans, 72, 27, 101, 103, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::nontrans, 72, 27, 101, 103, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::lower,
        oneapi::mkl::transpose::conjtrans, 72, 27, 101, 103, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::conjtrans, 72, 27, 101, 103, alpha, beta)));
}

INSTANTIATE_TEST_SUITE_P(HerkTestSuite, HerkTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::column_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
