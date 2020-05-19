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
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include <CL/sycl.hpp>
#include "allocator_helper.hpp"
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
         onemkl::transpose transb, int n, int k, int lda, int ldb, int ldc, fp alpha, fp beta) {
    // Prepare data.
    vector<fp, allocator_helper<fp, 64>> A, B, C, C_ref;
    rand_matrix(A, transa, n, k, lda);
    rand_matrix(B, transb, k, n, ldb);
    rand_matrix(C, onemkl::transpose::nontrans, n, n, ldc);
    C_ref = C;

    // Call Reference GEMMT.
    const int n_ref = n, k_ref = k;
    const int lda_ref = lda, ldb_ref = ldb, ldc_ref = ldc;

    using fp_ref = typename ref_type_info<fp>::type;

    ::gemmt(convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
            convert_to_cblas_trans(transb), &n_ref, &k_ref, (fp_ref*)&alpha, (fp_ref*)A.data(),
            &lda_ref, (fp_ref*)B.data(), &ldb_ref, (fp_ref*)&beta, (fp_ref*)C_ref.data(), &ldc_ref);

    // Call DPC++ GEMMT.

    // Catch asynchronous exceptions
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMMT:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(dev, exception_handler);

    buffer<fp, 1> A_buffer(A.data(), range<1>(A.size()));
    buffer<fp, 1> B_buffer(B.data(), range<1>(B.size()));
    buffer<fp, 1> C_buffer(C.data(), range<1>(C.size()));

    try {
#ifdef CALL_RT_API
        onemkl::blas::gemmt(main_queue, upper_lower, transa, transb, n, k, alpha, A_buffer, lda,
                            B_buffer, ldb, beta, C_buffer, ldc);
#else
        TEST_RUN_CT(main_queue, onemkl::blas::gemmt,
                    (main_queue, upper_lower, transa, transb, n, k, alpha, A_buffer, lda, B_buffer,
                     ldb, beta, C_buffer, ldc));
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during GEMMT:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const onemkl::backend_unsupported_exception& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of GEMMT:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    auto C_accessor = C_buffer.template get_access<access::mode::read>();
    bool good = check_equal_matrix(C_accessor, C_ref, upper_lower, n, n, ldc, 10 * k, std::cout);

    return (int)good;
}

class GemmtTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(GemmtTests, RealSinglePrecision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans,
                                  onemkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans,
                                  onemkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::lower, onemkl::transpose::trans,
                                  onemkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::lower, onemkl::transpose::trans,
                                  onemkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans,
                                  onemkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans,
                                  onemkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::upper, onemkl::transpose::trans,
                                  onemkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::uplo::upper, onemkl::transpose::trans,
                                  onemkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
}

TEST_P(GemmtTests, RealDoublePrecision) {
    double alpha(2.0);
    double beta(3.0);
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans,
                                   onemkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha,
                                   beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans,
                                   onemkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::lower, onemkl::transpose::trans,
                                   onemkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha,
                                   beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::lower, onemkl::transpose::trans,
                                   onemkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans,
                                   onemkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha,
                                   beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans,
                                   onemkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::upper, onemkl::transpose::trans,
                                   onemkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha,
                                   beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::uplo::upper, onemkl::transpose::trans,
                                   onemkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
}

TEST_P(GemmtTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0);
    std::complex<float> beta(3.0);
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans,
                                  onemkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans,
                                  onemkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), onemkl::uplo::lower, onemkl::transpose::trans,
                                  onemkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::uplo::lower,
                                                onemkl::transpose::trans, onemkl::transpose::trans,
                                                27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans, onemkl::transpose::conjtrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::trans, onemkl::transpose::conjtrans, 27,
        98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), onemkl::uplo::lower, onemkl::transpose::conjtrans,
                                  onemkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), onemkl::uplo::lower, onemkl::transpose::conjtrans,
                                  onemkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::conjtrans, onemkl::transpose::conjtrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans,
                                  onemkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans,
                                  onemkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), onemkl::uplo::upper, onemkl::transpose::trans,
                                  onemkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::uplo::upper,
                                                onemkl::transpose::trans, onemkl::transpose::trans,
                                                27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans, onemkl::transpose::conjtrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::trans, onemkl::transpose::conjtrans, 27,
        98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), onemkl::uplo::upper, onemkl::transpose::conjtrans,
                                  onemkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), onemkl::uplo::upper, onemkl::transpose::conjtrans,
                                  onemkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::conjtrans, onemkl::transpose::conjtrans,
        27, 98, 101, 102, 103, alpha, beta));
}

TEST_P(GemmtTests, ComplexDoublePrecision) {
    std::complex<double> alpha(2.0);
    std::complex<double> beta(3.0);
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans, onemkl::transpose::nontrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans,
                                   onemkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::trans, onemkl::transpose::nontrans, 27,
        98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::uplo::lower,
                                                 onemkl::transpose::trans, onemkl::transpose::trans,
                                                 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::nontrans, onemkl::transpose::conjtrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::trans, onemkl::transpose::conjtrans, 27,
        98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::conjtrans, onemkl::transpose::nontrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(GetParam(), onemkl::uplo::lower, onemkl::transpose::conjtrans,
                                   onemkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::lower, onemkl::transpose::conjtrans, onemkl::transpose::conjtrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans, onemkl::transpose::nontrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans,
                                   onemkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::trans, onemkl::transpose::nontrans, 27,
        98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::uplo::upper,
                                                 onemkl::transpose::trans, onemkl::transpose::trans,
                                                 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::nontrans, onemkl::transpose::conjtrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::trans, onemkl::transpose::conjtrans, 27,
        98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::conjtrans, onemkl::transpose::nontrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(GetParam(), onemkl::uplo::upper, onemkl::transpose::conjtrans,
                                   onemkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), onemkl::uplo::upper, onemkl::transpose::conjtrans, onemkl::transpose::conjtrans,
        27, 98, 101, 102, 103, alpha, beta));
}

INSTANTIATE_TEST_SUITE_P(GemmtTestSuite, GemmtTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
