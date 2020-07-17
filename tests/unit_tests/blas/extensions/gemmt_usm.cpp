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

extern std::vector<cl::sycl::device> devices;

namespace {

template <typename fp>
int test(const device& dev, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
         oneapi::mkl::transpose transb, int n, int k, int lda, int ldb, int ldc, fp alpha, fp beta) {
    // Catch asynchronous exceptions.
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
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.
    auto ua = usm_allocator<fp, usm::alloc::shared, 64>(cxt, dev);
    vector<fp, decltype(ua)> A(ua), B(ua), C(ua);
    rand_matrix(A, transa, n, k, lda);
    rand_matrix(B, transb, k, n, ldb);
    rand_matrix(C, oneapi::mkl::transpose::nontrans, n, n, ldc);

    auto C_ref = C;

    // Call Reference GEMMT.
    const int n_ref = n, k_ref = k;
    const int lda_ref = lda, ldb_ref = ldb, ldc_ref = ldc;

    using fp_ref = typename ref_type_info<fp>::type;

    ::gemmt(convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
            convert_to_cblas_trans(transb), &n_ref, &k_ref, (fp_ref*)&alpha, (fp_ref*)A.data(),
            &lda_ref, (fp_ref*)B.data(), &ldb_ref, (fp_ref*)&beta, (fp_ref*)C_ref.data(), &ldc_ref);

    // Call DPC++ GEMMT.

    try {
#ifdef CALL_RT_API
        done = oneapi::mkl::blas::gemmt(main_queue, upper_lower, transa, transb, n, k, alpha, A.data(),
                                   lda, B.data(), ldb, beta, C.data(), ldc, dependencies);
        done.wait();
#else
        TEST_RUN_CT(main_queue, oneapi::mkl::blas::gemmt,
                    (main_queue, upper_lower, transa, transb, n, k, alpha, A.data(), lda, B.data(),
                     ldb, beta, C.data(), ldc, dependencies));
        main_queue.wait();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during GEMMT:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const oneapi::mkl::backend_unsupported_exception& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of GEMMT:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good = check_equal_matrix(C, C_ref, upper_lower, n, n, ldc, 10 * k, std::cout);

    return (int)good;
}

class GemmtUsmTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(GemmtUsmTests, RealSinglePrecision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUEORSKIP(test<float>(GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
                                  oneapi::mkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
                                  oneapi::mkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::trans,
                                  oneapi::mkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::trans,
                                  oneapi::mkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans,
                                  oneapi::mkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans,
                                  oneapi::mkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans,
                                  oneapi::mkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans,
                                  oneapi::mkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
}

TEST_P(GemmtUsmTests, RealDoublePrecision) {
    double alpha(2.0);
    double beta(3.0);
    EXPECT_TRUEORSKIP(test<double>(GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
                                   oneapi::mkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha,
                                   beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
                                   oneapi::mkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::trans,
                                   oneapi::mkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha,
                                   beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::trans,
                                   oneapi::mkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans,
                                   oneapi::mkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha,
                                   beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans,
                                   oneapi::mkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans,
                                   oneapi::mkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha,
                                   beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans,
                                   oneapi::mkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
}

TEST_P(GemmtUsmTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0);
    std::complex<float> beta(3.0);
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
                                  oneapi::mkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
                                  oneapi::mkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::trans,
                                  oneapi::mkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), oneapi::mkl::uplo::lower,
                                                oneapi::mkl::transpose::trans, oneapi::mkl::transpose::trans,
                                                27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::conjtrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::conjtrans, 27,
        98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::conjtrans,
                                  oneapi::mkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::conjtrans,
                                  oneapi::mkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans,
                                  oneapi::mkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans,
                                  oneapi::mkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans,
                                  oneapi::mkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), oneapi::mkl::uplo::upper,
                                                oneapi::mkl::transpose::trans, oneapi::mkl::transpose::trans,
                                                27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::conjtrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::conjtrans, 27,
        98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans,
                                  oneapi::mkl::transpose::nontrans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans,
                                  oneapi::mkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans,
        27, 98, 101, 102, 103, alpha, beta));
}

TEST_P(GemmtUsmTests, ComplexDoublePrecision) {
    std::complex<double> alpha(2.0);
    std::complex<double> beta(3.0);
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
                                   oneapi::mkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, 27,
        98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), oneapi::mkl::uplo::lower,
                                                 oneapi::mkl::transpose::trans, oneapi::mkl::transpose::trans,
                                                 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::conjtrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::conjtrans, 27,
        98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::nontrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::conjtrans,
                                   oneapi::mkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), oneapi::mkl::uplo::lower, oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans,
                                   oneapi::mkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, 27,
        98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), oneapi::mkl::uplo::upper,
                                                 oneapi::mkl::transpose::trans, oneapi::mkl::transpose::trans,
                                                 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::conjtrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::conjtrans, 27,
        98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::nontrans,
        27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans,
                                   oneapi::mkl::transpose::trans, 27, 98, 101, 102, 103, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        GetParam(), oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::conjtrans,
        27, 98, 101, 102, 103, alpha, beta));
}

INSTANTIATE_TEST_SUITE_P(GemmtUsmTestSuite, GemmtUsmTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
