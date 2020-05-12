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

template <typename Ta, typename Tc>
bool test(const device& dev, onemkl::transpose transa, onemkl::transpose transb, int m, int n,
          int k, int lda, int ldb, int ldc, Tc alpha, Tc beta) {
    // Prepare data.
    vector<Ta, allocator_helper<Ta, 64>> A, B;
    vector<Tc, allocator_helper<Tc, 64>> C, C_ref;
    rand_matrix(A, transa, m, k, lda);
    rand_matrix(B, transb, k, n, ldb);
    rand_matrix(C, onemkl::transpose::nontrans, m, n, ldc);
    C_ref = C;

    // Call Reference GEMM_EXT.
    const int m_ref = m, n_ref = n, k_ref = k;
    const int lda_ref = lda, ldb_ref = ldb, ldc_ref = ldc;

    using Ta_ref = typename ref_type_info<Ta>::type;
    using Tc_ref = typename ref_type_info<Tc>::type;

    ::gemm_ext(convert_to_cblas_trans(transa), convert_to_cblas_trans(transb), &m_ref, &n_ref,
               &k_ref, (Tc_ref*)&alpha, (Ta_ref*)A.data(), &lda_ref, (Ta_ref*)B.data(), &ldb_ref,
               (Tc_ref*)&beta, (Tc_ref*)C_ref.data(), &ldc_ref);

    // Call DPC++ GEMM_EXT.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMM_EXT:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(dev, exception_handler);

    buffer<Ta, 1> A_buffer(A.data(), range<1>(A.size()));
    buffer<Ta, 1> B_buffer(B.data(), range<1>(B.size()));
    buffer<Tc, 1> C_buffer(C.data(), range<1>(C.size()));

    try {
#ifdef CALL_RT_API
        onemkl::blas::gemm_ext(main_queue, transa, transb, m, n, k, alpha, A_buffer, lda, B_buffer,
                               ldb, beta, C_buffer, ldc);
#else
        TEST_RUN_CT(main_queue, onemkl::blas::gemm_ext,
                    (main_queue, transa, transb, m, n, k, alpha, A_buffer, lda, B_buffer, ldb, beta,
                     C_buffer, ldc));
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during GEMM_EXT:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of GEMM_EXT:\n" << error.what() << std::endl;
#ifdef ENABLE_CUBLAS_BACKEND
        // GEMM_EXT currently not supported with CUBLAS backend.
        std::string error_msg(error.what());
        if (error_msg.compare("Not implemented for cublas") == 0) {
            return true;
        }
#endif
    }

    // Compare the results of reference implementation and DPC++ implementation.
    auto C_accessor = C_buffer.template get_access<access::mode::read>();
    bool good       = check_equal_matrix(C_accessor, C_ref, m, n, ldc, 10 * k, std::cout);

    return good;
}

class GemmExtTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(GemmExtTests, HalfHalfFloatPrecision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUE(
        (test<half, float>(GetParam(), onemkl::transpose::nontrans, onemkl::transpose::nontrans, 79,
                           83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUE(
        (test<half, float>(GetParam(), onemkl::transpose::nontrans, onemkl::transpose::trans, 79,
                           83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUE(
        (test<half, float>(GetParam(), onemkl::transpose::trans, onemkl::transpose::nontrans, 79,
                           83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUE((test<half, float>(GetParam(), onemkl::transpose::trans, onemkl::transpose::trans,
                                   79, 83, 91, 103, 105, 106, alpha, beta)));
}

TEST_P(GemmExtTests, RealHalfPrecision) {
    half alpha(2.0);
    half beta(3.0);
    EXPECT_TRUE(
        (test<half, half>(GetParam(), onemkl::transpose::nontrans, onemkl::transpose::nontrans, 79,
                          83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUE((test<half, half>(GetParam(), onemkl::transpose::nontrans, onemkl::transpose::trans,
                                  79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUE((test<half, half>(GetParam(), onemkl::transpose::trans, onemkl::transpose::nontrans,
                                  79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUE((test<half, half>(GetParam(), onemkl::transpose::trans, onemkl::transpose::trans,
                                  79, 83, 91, 103, 105, 106, alpha, beta)));
}

TEST_P(GemmExtTests, RealSinglePrecision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUE(
        (test<float, float>(GetParam(), onemkl::transpose::nontrans, onemkl::transpose::nontrans,
                            79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUE(
        (test<float, float>(GetParam(), onemkl::transpose::nontrans, onemkl::transpose::trans, 79,
                            83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUE(
        (test<float, float>(GetParam(), onemkl::transpose::trans, onemkl::transpose::nontrans, 79,
                            83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUE((test<float, float>(GetParam(), onemkl::transpose::trans, onemkl::transpose::trans,
                                    79, 83, 91, 103, 105, 106, alpha, beta)));
}

TEST_P(GemmExtTests, RealDoublePrecision) {
    double alpha(2.0);
    double beta(3.0);
    EXPECT_TRUE(
        (test<double, double>(GetParam(), onemkl::transpose::nontrans, onemkl::transpose::nontrans,
                              79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUE(
        (test<double, double>(GetParam(), onemkl::transpose::nontrans, onemkl::transpose::trans, 79,
                              83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUE(
        (test<double, double>(GetParam(), onemkl::transpose::trans, onemkl::transpose::nontrans, 79,
                              83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUE(
        (test<double, double>(GetParam(), onemkl::transpose::trans, onemkl::transpose::trans, 79,
                              83, 91, 103, 105, 106, alpha, beta)));
}

TEST_P(GemmExtTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0, -0.5);
    std::complex<float> beta(3.0, -1.5);
    EXPECT_TRUE((test<std::complex<float>, std::complex<float>>(
        GetParam(), onemkl::transpose::nontrans, onemkl::transpose::nontrans, 79, 83, 91, 103, 105,
        106, alpha, beta)));
    EXPECT_TRUE((test<std::complex<float>, std::complex<float>>(
        GetParam(), onemkl::transpose::nontrans, onemkl::transpose::trans, 79, 83, 91, 103, 105,
        106, alpha, beta)));
    EXPECT_TRUE((test<std::complex<float>, std::complex<float>>(
        GetParam(), onemkl::transpose::trans, onemkl::transpose::nontrans, 79, 83, 91, 103, 105,
        106, alpha, beta)));
    EXPECT_TRUE((test<std::complex<float>, std::complex<float>>(
        GetParam(), onemkl::transpose::trans, onemkl::transpose::trans, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUE((test<std::complex<float>, std::complex<float>>(
        GetParam(), onemkl::transpose::nontrans, onemkl::transpose::conjtrans, 79, 83, 91, 103, 105,
        106, alpha, beta)));
    EXPECT_TRUE((test<std::complex<float>, std::complex<float>>(
        GetParam(), onemkl::transpose::trans, onemkl::transpose::conjtrans, 79, 83, 91, 103, 105,
        106, alpha, beta)));
    EXPECT_TRUE((test<std::complex<float>, std::complex<float>>(
        GetParam(), onemkl::transpose::conjtrans, onemkl::transpose::nontrans, 79, 83, 91, 103, 105,
        106, alpha, beta)));
    EXPECT_TRUE((test<std::complex<float>, std::complex<float>>(
        GetParam(), onemkl::transpose::conjtrans, onemkl::transpose::trans, 79, 83, 91, 103, 105,
        106, alpha, beta)));
    EXPECT_TRUE((test<std::complex<float>, std::complex<float>>(
        GetParam(), onemkl::transpose::conjtrans, onemkl::transpose::conjtrans, 79, 83, 91, 103,
        105, 106, alpha, beta)));
}

TEST_P(GemmExtTests, ComplexDoublePrecision) {
    std::complex<double> alpha(2.0, -0.5);
    std::complex<double> beta(3.0, -1.5);
    EXPECT_TRUE((test<std::complex<double>, std::complex<double>>(
        GetParam(), onemkl::transpose::nontrans, onemkl::transpose::nontrans, 79, 83, 91, 103, 105,
        106, alpha, beta)));
    EXPECT_TRUE((test<std::complex<double>, std::complex<double>>(
        GetParam(), onemkl::transpose::nontrans, onemkl::transpose::trans, 79, 83, 91, 103, 105,
        106, alpha, beta)));
    EXPECT_TRUE((test<std::complex<double>, std::complex<double>>(
        GetParam(), onemkl::transpose::trans, onemkl::transpose::nontrans, 79, 83, 91, 103, 105,
        106, alpha, beta)));
    EXPECT_TRUE((test<std::complex<double>, std::complex<double>>(
        GetParam(), onemkl::transpose::trans, onemkl::transpose::trans, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUE((test<std::complex<double>, std::complex<double>>(
        GetParam(), onemkl::transpose::nontrans, onemkl::transpose::conjtrans, 79, 83, 91, 103, 105,
        106, alpha, beta)));
    EXPECT_TRUE((test<std::complex<double>, std::complex<double>>(
        GetParam(), onemkl::transpose::trans, onemkl::transpose::conjtrans, 79, 83, 91, 103, 105,
        106, alpha, beta)));
    EXPECT_TRUE((test<std::complex<double>, std::complex<double>>(
        GetParam(), onemkl::transpose::conjtrans, onemkl::transpose::nontrans, 79, 83, 91, 103, 105,
        106, alpha, beta)));
    EXPECT_TRUE((test<std::complex<double>, std::complex<double>>(
        GetParam(), onemkl::transpose::conjtrans, onemkl::transpose::trans, 79, 83, 91, 103, 105,
        106, alpha, beta)));
    EXPECT_TRUE((test<std::complex<double>, std::complex<double>>(
        GetParam(), onemkl::transpose::conjtrans, onemkl::transpose::conjtrans, 79, 83, 91, 103,
        105, 106, alpha, beta)));
}

INSTANTIATE_TEST_SUITE_P(GemmExtTestSuite, GemmExtTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
